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
        # original_tid -> {"box": (x1, y1, x2, y2), "center": (cx, cy), "class_name": str}
        self.counted_spatial_history: Dict[int, Dict] = {}
        # Alias map: new_flickered_tid -> original_counted_tid
        self.id_aliases: Dict[int, int] = {}

    def process(self, tracked_objects: Iterable[dict]) -> Dict[str, int]:
        new_counts: DefaultDict[str, int] = defaultdict(int)
        if not tracked_objects:
            # Increment misses for all pending tracks
            pending_ids = [tid for tid in list(self.track_hits.keys()) if tid not in self.global_counted_track_ids]
            for tid in pending_ids:
                self.track_misses[tid] += 1
                if self.track_misses[tid] >= 6:
                    self.track_hits.pop(tid, None)
                    self.track_misses.pop(tid, None)
                    self.track_first_pos.pop(tid, None)
            return dict(new_counts)

        present_ids: Set[int] = set()

        for item in tracked_objects:
            raw_tid = item.get("tracking_id")
            raw_class_name = item.get("class_name", "Unknown")

            if raw_tid is None:
                item["is_counted"] = False
                item["is_new"] = False
                continue

            # Resolve known alias if this ID was previously matched to a counted object
            tracking_id = self.id_aliases.get(raw_tid, raw_tid)
            item["tracking_id"] = tracking_id

            has_box = "x1" in item and "y1" in item and "x2" in item and "y2" in item
            if has_box:
                curr_box = (float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"]))
                cx = (curr_box[0] + curr_box[2]) / 2.0
                cy = (curr_box[1] + curr_box[3]) / 2.0
                bw = curr_box[2] - curr_box[0]
                bh = curr_box[3] - curr_box[1]
                diag = math.hypot(bw, bh)
            else:
                curr_box = (0.0, 0.0, 0.0, 0.0)
                cx, cy, diag = 0.0, 0.0, 0.0

            present_ids.add(tracking_id)

            # CASE A: Track ID is already in the global counted set
            if tracking_id in self.global_counted_track_ids:
                item["class_name"] = self.track_class_map.get(tracking_id, raw_class_name)
                item["is_counted"] = True
                item["is_new"] = False
                if has_box and tracking_id in self.counted_spatial_history:
                    self.counted_spatial_history[tracking_id]["box"] = curr_box
                    self.counted_spatial_history[tracking_id]["center"] = (cx, cy)
                continue

            # CASE B: Track ID is NOT yet marked counted, but check if it spatially belongs
            # to an already-counted object (anti-duplicate Re-ID)
            matched_counted_id = None
            if has_box and self.counted_spatial_history:
                for c_id, c_data in self.counted_spatial_history.items():
                    c_box = c_data["box"]
                    c_cx, c_cy = c_data["center"]
                    c_diag = math.hypot(c_box[2] - c_box[0], c_box[3] - c_box[1])
                    avg_diag = max(20.0, (diag + c_diag) / 2.0)
                    
                    iou = _calc_iou(curr_box, c_box)
                    dist = math.hypot(cx - c_cx, cy - c_cy)

                    # Match if significant IoU or center is within 1.2x object diagonal
                    if iou > 0.15 or (iou > 0.05 and dist < avg_diag * 0.9) or (dist < avg_diag * 0.65):
                        matched_counted_id = c_id
                        break

            if matched_counted_id is not None:
                # Link this flickered track ID to the already-counted physical object
                self.id_aliases[raw_tid] = matched_counted_id
                item["tracking_id"] = matched_counted_id
                item["class_name"] = self.track_class_map.get(matched_counted_id, raw_class_name)
                item["is_counted"] = True
                item["is_new"] = False
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
            # Require at least 4 stable hits (or 3 hits with noticeable flow motion)
            is_confirmed = (hits >= self.min_hits and dist_moved >= 12.0) or (hits >= (self.min_hits + 1))

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
        canonical_id = self.id_aliases.get(tracking_id, tracking_id)
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

    def get_unique_track_ids(self) -> Dict[str, set]:
        res = defaultdict(set)
        for tid, cls_name in self.track_class_map.items():
            res[cls_name].add(tid)
        return dict(res)


