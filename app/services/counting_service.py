from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


class CountingService:
    def __init__(self, min_hits: int = 2):
        """
        Counting service with confirmation threshold (min_hits).
        Requiring min_hits >= 2 ensures momentary false detections (glitches/shadows/hand motion)
        are ignored, and once a unique tracking_id is counted, it can NEVER be counted again.
        """
        self.min_hits = min_hits
        # Maps tracking_id -> consecutive appearance count
        self.track_hits: Dict[int, int] = defaultdict(int)
        # Global set of tracking_ids that have been confirmed & counted: A physical object is counted AT MOST ONCE!
        self.global_counted_track_ids: Set[int] = set()
        # Maps tracking_id -> confirmed class_name (locks class identity)
        self.track_class_map: Dict[int, str] = {}
        # Accumulated count per class
        self.accumulated_by_class: DefaultDict[str, int] = defaultdict(int)

    def process(self, tracked_objects: Iterable[dict]) -> Dict[str, int]:
        new_counts: DefaultDict[str, int] = defaultdict(int)
        if not tracked_objects:
            return dict(new_counts)

        for item in tracked_objects:
            tracking_id = item.get("tracking_id")
            raw_class_name = item.get("class_name", "Unknown")

            if tracking_id is None:
                item["is_counted"] = False
                item["is_new"] = False
                continue

            # Check if this physical object was already counted previously
            if tracking_id in self.global_counted_track_ids:
                # Lock and maintain the confirmed class name
                item["class_name"] = self.track_class_map.get(tracking_id, raw_class_name)
                item["is_counted"] = True
                item["is_new"] = False
            else:
                self.track_hits[tracking_id] += 1
                if self.track_hits[tracking_id] >= self.min_hits:
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
        self.global_counted_track_ids.clear()
        self.track_class_map.clear()
        self.accumulated_by_class.clear()

    def get_unique_track_ids(self) -> Dict[str, set]:
        res = defaultdict(set)
        for tid, cls_name in self.track_class_map.items():
            res[cls_name].add(tid)
        return dict(res)
