import math
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


def _calc_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class CountingService:
    def __init__(self, min_hits: int = 2):
        """
        Anti-duplicate precision counting service:
        - Confirmed objects are counted EXACTLY ONCE.
        - Stores spatial memory of all counted objects to prevent re-counting
          when tracker IDs switch, flicker, or drop.
        - Uses Co-Presence Exclusion to prevent adjacent objects in the same frame
          from falsely merging as spatial aliases.
        """
        self.min_hits = min_hits
        # Maps tracking_id -> appearance hit count
        self.track_hits: Dict[int, int] = defaultdict(int)
        # Maps tracking_id -> consecutive unseen frames
        self.track_misses: Dict[int, int] = defaultdict(int)
        # First seen center coordinates (cx, cy)
        self.track_first_pos: Dict[int, Tuple[float, float]] = {}
        # Global set of physical tracking_ids that have been confirmed & counted
        self.global_counted_track_ids: Set[int] = set()
        # Maps tracking_id -> confirmed class_name
        self.track_class_map: Dict[int, str] = {}
        # Accumulated count per class
        self.accumulated_by_class: DefaultDict[str, int] = defaultdict(int)

        # Spatial memory of all counted objects:
        # original_tid -> {"box": (x1, y1, x2, y2), "center": (cx, cy), "class_name": str, "last_seen_frame": int}
        self.counted_spatial_history: Dict[int, Dict] = {}
        # Alias map: new_flickered_tid -> original_counted_tid
        self.id_aliases: Dict[int, int] = {}
        # Current video frame counter
        self.current_frame: int = 0

    def get_canonical_id(self, tracking_id: int) -> int:
        """Resolve canonical tracking ID through transitive aliases."""
        visited: Set[int] = set()
        curr = tracking_id
        while curr in self.id_aliases and curr not in visited:
            visited.add(curr)
            curr = self.id_aliases[curr]
        return curr

    def update(self, active_detections: Iterable[dict], frame_number: Optional[int] = None) -> Dict[str, int]:
        """
        Public interface contract: update(active_detections, frame_number) -> Dict[str, int].
        Updates active_detections in-place and returns new_counts by class.
        """
        return self.process(active_detections, frame_number=frame_number)

    def process(self, tracked_objects: Iterable[dict], frame_number: Optional[int] = None) -> Dict[str, int]:
        new_counts: DefaultDict[str, int] = defaultdict(int)
        if tracked_objects is None:
            tracked_list = []
        elif isinstance(tracked_objects, list):
            tracked_list = tracked_objects
        else:
            tracked_list = list(tracked_objects)

        # Advance / set frame number
        if frame_number is not None:
            self.current_frame = frame_number
        else:
            frame_candidates = [
                item["frame_number"]
                for item in tracked_list
                if isinstance(item, dict) and item.get("frame_number") is not None
            ]
            if frame_candidates:
                self.current_frame = max(frame_candidates)
            else:
                self.current_frame += 1

        if not tracked_list:
            # Increment misses for all pending tracks
            pending_ids = [tid for tid in list(self.track_hits.keys()) if tid not in self.global_counted_track_ids]
            for tid in pending_ids:
                self.track_misses[tid] += 1
                if self.track_misses[tid] >= 6:
                    self.track_hits.pop(tid, None)
                    self.track_misses.pop(tid, None)
                    self.track_first_pos.pop(tid, None)
            return dict(new_counts)

        # Filter out composite cluster boxes (boxes enclosing other tracked items in the same frame)
        # and full-screen background false positives (> 50% frame area)
        composite_indices: Set[int] = set()
        for i, it1 in enumerate(tracked_list):
            if "x1" not in it1 or "x2" not in it1:
                continue
            b1 = (float(it1["x1"]), float(it1["y1"]), float(it1["x2"]), float(it1["y2"]))
            bw1 = max(0.0, b1[2] - b1[0])
            bh1 = max(0.0, b1[3] - b1[1])
            a1 = bw1 * bh1
            # Full-screen / giant background box rejection
            if bw1 > 1000 or (bw1 > 700 and bh1 > 280) or a1 > 500000:
                composite_indices.add(i)
                continue
            for j, it2 in enumerate(tracked_list):
                if i == j:
                    continue
                if "x1" not in it2 or "x2" not in it2:
                    continue
                b2 = (float(it2["x1"]), float(it2["y1"]), float(it2["x2"]), float(it2["y2"]))
                a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
                if a1 > 1.6 * a2:
                    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    if (inter / a2) > 0.45:
                        composite_indices.add(i)
                        break

        filtered_items = [it for idx, it in enumerate(tracked_list) if idx not in composite_indices]

        # Collect all raw tracking IDs currently visible in this frame
        current_frame_raw_tids: Set[int] = {
            item.get("tracking_id")
            for item in filtered_items
            if isinstance(item, dict) and item.get("tracking_id") is not None
        }

        present_ids: Set[int] = set()

        for item in filtered_items:
            raw_tid = item.get("tracking_id")
            raw_class_name = item.get("class_name", "Unknown")

            if raw_tid is None:
                item["is_counted"] = False
                item["is_new"] = False
                continue

            # Resolve known alias if this ID was previously matched to a counted object
            tracking_id = self.get_canonical_id(raw_tid)
            item["tracking_id"] = tracking_id

            # Parse bounding box robustly
            if "box" in item and isinstance(item["box"], (list, tuple)) and len(item["box"]) >= 4:
                curr_box = (float(item["box"][0]), float(item["box"][1]), float(item["box"][2]), float(item["box"][3]))
                has_box = True
            elif "x1" in item and "y1" in item and "x2" in item and "y2" in item:
                curr_box = (float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"]))
                has_box = True
            else:
                curr_box = (0.0, 0.0, 0.0, 0.0)
                has_box = False

            if has_box:
                cx = (curr_box[0] + curr_box[2]) / 2.0
                cy = (curr_box[1] + curr_box[3]) / 2.0
                bw = max(1.0, curr_box[2] - curr_box[0])
                bh = max(1.0, curr_box[3] - curr_box[1])
                diag = math.hypot(bw, bh)
            else:
                cx, cy, diag = 0.0, 0.0, 0.0

            present_ids.add(raw_tid)
            present_ids.add(tracking_id)

            # CASE A: Track ID is already in the global counted set
            if tracking_id in self.global_counted_track_ids:
                item["class_name"] = self.track_class_map.get(tracking_id, raw_class_name)
                item["is_counted"] = True
                item["is_new"] = False
                if tracking_id in self.counted_spatial_history:
                    self.counted_spatial_history[tracking_id]["last_seen_frame"] = self.current_frame
                    if has_box:
                        self.counted_spatial_history[tracking_id]["box"] = curr_box
                        self.counted_spatial_history[tracking_id]["center"] = (cx, cy)
                continue

            # CASE B: Track ID is NOT yet marked counted. Check if it spatially belongs
            # to an already-counted object (anti-duplicate Re-ID)
            matched_counted_id = None
            if has_box and self.counted_spatial_history:
                # Co-Presence Exclusion: Identify all counted records active in this frame
                # via OTHER detections. Two physically distinct objects co-present in the
                # same frame can NEVER be aliases of each other.
                other_frame_tids = current_frame_raw_tids - {raw_tid}
                active_counted_in_frame: Set[int] = set()
                for otid in other_frame_tids:
                    active_counted_in_frame.add(otid)
                    o_canon = self.get_canonical_id(otid)
                    active_counted_in_frame.add(o_canon)

                best_match_id = None
                best_score = -1.0

                for c_id, c_data in self.counted_spatial_history.items():
                    # Co-Presence Exclusion
                    if c_id in active_counted_in_frame:
                        continue

                    # Frame-based aging: only consider recent history within recovery window (<= 60 frames)
                    last_seen = c_data.get("last_seen_frame", self.current_frame)
                    if (self.current_frame - last_seen) > 60:
                        continue

                    c_box = c_data["box"]
                    c_cx, c_cy = c_data["center"]
                    c_diag = math.hypot(c_box[2] - c_box[0], c_box[3] - c_box[1])
                    min_diag = min(diag, c_diag)

                    iou = _calc_iou(curr_box, c_box)
                    dist = math.hypot(cx - c_cx, cy - c_cy)

                    # IoU-gated condition: Require substantial overlap or moderate overlap with close proximity
                    if iou > 0.25 or (iou > 0.10 and dist < min_diag * 0.40):
                        score = iou * 10.0 - (dist / max(1.0, min_diag))
                        if score > best_score:
                            best_score = score
                            best_match_id = c_id

                matched_counted_id = best_match_id

            if matched_counted_id is not None:
                # Link this flickered track ID to the already-counted physical object
                self.id_aliases[raw_tid] = matched_counted_id
                item["tracking_id"] = matched_counted_id
                item["class_name"] = self.track_class_map.get(matched_counted_id, raw_class_name)
                item["is_counted"] = True
                item["is_new"] = False
                # Update spatial history to current box to maintain tracker continuity
                self.counted_spatial_history[matched_counted_id]["last_seen_frame"] = self.current_frame
                if has_box:
                    self.counted_spatial_history[matched_counted_id]["box"] = curr_box
                    self.counted_spatial_history[matched_counted_id]["center"] = (cx, cy)
                continue

            # CASE C: Candidate new physical object - accumulate hits
            self.track_hits[tracking_id] += 1
            self.track_misses[tracking_id] = 0

            if tracking_id not in self.track_first_pos and has_box:
                self.track_first_pos[tracking_id] = (cx, cy)

            init_cx, init_cy = self.track_first_pos.get(tracking_id, (cx, cy))
            dist_moved = math.hypot(cx - init_cx, cy - init_cy)
            hits = self.track_hits[tracking_id]

            # Confirmation threshold:
            # Static cans (standing on a table or in a multi-pack) with hits >= min_hits are confirmed cleanly.
            # Moving objects with noticeable flow motion (hits >= 2 and dist_moved >= 10.0) can also confirm.
            is_confirmed = (hits >= self.min_hits) or (hits >= 2 and dist_moved >= 10.0)

            if is_confirmed:
                # Confirmed object! Add to official count ONCE
                self.global_counted_track_ids.add(tracking_id)
                self.track_class_map[tracking_id] = raw_class_name
                self.accumulated_by_class[raw_class_name] += 1
                if has_box:
                    self.counted_spatial_history[tracking_id] = {
                        "box": curr_box,
                        "center": (cx, cy),
                        "class_name": raw_class_name,
                        "last_seen_frame": self.current_frame,
                    }
                item["is_counted"] = True
                item["is_new"] = True
                new_counts[raw_class_name] += 1
            else:
                item["is_counted"] = False
                item["is_new"] = False

        # Cleanup unconfirmed tracks that disappeared for >= 5 consecutive frames
        pending_ids = [tid for tid in list(self.track_hits.keys()) if tid not in self.global_counted_track_ids]
        for tid in pending_ids:
            if tid not in present_ids:
                self.track_misses[tid] += 1
                if self.track_misses[tid] >= 5:
                    self.track_hits.pop(tid, None)
                    self.track_misses.pop(tid, None)
                    self.track_first_pos.pop(tid, None)

        return dict(new_counts)

    def is_already_counted(self, tracking_id: int) -> bool:
        canonical_id = self.get_canonical_id(tracking_id)
        return canonical_id in self.global_counted_track_ids

    def get_total(self, counts: Dict[str, int]) -> int:
        return sum(counts.values())

    def get_accumulated_counts(self) -> Dict[str, int]:
        return dict(self.accumulated_by_class)

    def get_total_accumulated(self) -> int:
        return len(self.global_counted_track_ids)

    def reset(self) -> None:
        self.track_hits.clear()
        self.track_misses.clear()
        self.track_first_pos.clear()
        self.global_counted_track_ids.clear()
        self.track_class_map.clear()
        self.accumulated_by_class.clear()
        self.counted_spatial_history.clear()
        self.id_aliases.clear()
        self.current_frame = 0

    def get_unique_track_ids(self) -> Dict[str, set]:
        res = defaultdict(set)
        for tid, cls_name in self.track_class_map.items():
            res[cls_name].add(tid)
        return dict(res)


