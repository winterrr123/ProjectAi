import math
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


class CountingService:
    def __init__(self, min_hits: int = 3):
        """
        Counting service with confirmation threshold (min_hits).
        Requiring min_hits >= 3 ensures momentary false detections (glitches/shadows/hand motion)
        are ignored, and once a unique tracking_id is counted, it can NEVER be counted again.
        """
        self.min_hits = min_hits
        # Maps tracking_id -> consecutive appearance count
        self.track_hits: Dict[int, int] = defaultdict(int)
        # Maps tracking_id -> consecutive unseen frame count for pending tracks
        self.track_misses: Dict[int, int] = defaultdict(int)
        # First seen center coordinates (cx, cy) to monitor movement trajectory
        self.track_first_pos: Dict[int, Tuple[float, float]] = {}
        # Global set of tracking_ids that have been confirmed & counted: A physical object is counted AT MOST ONCE!
        self.global_counted_track_ids: Set[int] = set()
        # Maps tracking_id -> confirmed class_name (locks class identity)
        self.track_class_map: Dict[int, str] = {}
        # Accumulated count per class
        self.accumulated_by_class: DefaultDict[str, int] = defaultdict(int)

    def process(self, tracked_objects: Iterable[dict]) -> Dict[str, int]:
        new_counts: DefaultDict[str, int] = defaultdict(int)
        if not tracked_objects:
            # Increment misses for all pending tracks
            pending_ids = [tid for tid in list(self.track_hits.keys()) if tid not in self.global_counted_track_ids]
            for tid in pending_ids:
                self.track_misses[tid] += 1
                if self.track_misses[tid] >= 4:
                    self.track_hits.pop(tid, None)
                    self.track_misses.pop(tid, None)
                    self.track_first_pos.pop(tid, None)
            return dict(new_counts)

        present_ids: Set[int] = set()

        for item in tracked_objects:
            tracking_id = item.get("tracking_id")
            raw_class_name = item.get("class_name", "Unknown")

            if tracking_id is None:
                item["is_counted"] = False
                item["is_new"] = False
                continue

            present_ids.add(tracking_id)

            # Check if this physical object was already counted previously
            if tracking_id in self.global_counted_track_ids:
                # Lock and maintain the confirmed class name
                item["class_name"] = self.track_class_map.get(tracking_id, raw_class_name)
                item["is_counted"] = True
                item["is_new"] = False
            else:
                self.track_hits[tracking_id] += 1
                self.track_misses[tracking_id] = 0

                # Compute center coordinate for trajectory validation
                if "x1" in item and "y1" in item and "x2" in item and "y2" in item:
                    cx = (item["x1"] + item["x2"]) / 2.0
                    cy = (item["y1"] + item["y2"]) / 2.0
                    if tracking_id not in self.track_first_pos:
                        self.track_first_pos[tracking_id] = (cx, cy)
                    init_cx, init_cy = self.track_first_pos[tracking_id]
                    dist_moved = math.hypot(cx - init_cx, cy - init_cy)
                else:
                    dist_moved = 0.0

                hits = self.track_hits[tracking_id]
                # Confirmation criteria:
                # 1. Has reached min_hits AND has physically moved (conveyor/flow motion >= 8px)
                # OR 2. Has sustained detection >= min_hits + 1 frames (stable presence)
                is_confirmed = (hits >= self.min_hits and dist_moved >= 8.0) or (hits >= (self.min_hits + 1))

                if is_confirmed:
                    # Confirmed object! Add to official count ONCE
                    self.global_counted_track_ids.add(tracking_id)
                    self.track_class_map[tracking_id] = raw_class_name
                    self.accumulated_by_class[raw_class_name] += 1
                    item["is_counted"] = True
                    item["is_new"] = True
                    new_counts[raw_class_name] += 1
                else:
                    # Pending confirmation
                    item["is_counted"] = False
                    item["is_new"] = False

        # Cleanup unconfirmed tracks that disappeared for >= 4 consecutive frames
        pending_ids = [tid for tid in list(self.track_hits.keys()) if tid not in self.global_counted_track_ids]
        for tid in pending_ids:
            if tid not in present_ids:
                self.track_misses[tid] += 1
                if self.track_misses[tid] >= 4:
                    self.track_hits.pop(tid, None)
                    self.track_misses.pop(tid, None)
                    self.track_first_pos.pop(tid, None)

        return dict(new_counts)

    def is_already_counted(self, tracking_id: int) -> bool:
        return tracking_id in self.global_counted_track_ids

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

    def get_unique_track_ids(self) -> Dict[str, set]:
        res = defaultdict(set)
        for tid, cls_name in self.track_class_map.items():
            res[cls_name].add(tid)
        return dict(res)

