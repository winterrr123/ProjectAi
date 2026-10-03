from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from app.models.tracking import TrackedObject


def calculate_iou(box1: tuple, box2: tuple) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0.0 else 0.0


def calculate_center_distance(box1: tuple, box2: tuple) -> float:
    """Calculate Euclidean distance between centers of two boxes."""
    c1x = (box1[0] + box1[2]) / 2.0
    c1y = (box1[1] + box1[3]) / 2.0
    c2x = (box2[0] + box2[2]) / 2.0
    c2y = (box2[1] + box2[3]) / 2.0
    return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5


class ByteTrackService:
    def __init__(self, max_lost_age: int = 60):
        # Store active tracks: tracking_id -> TrackedObject (Global spatial tracking)
        self.track_store: Dict[int, TrackedObject] = {}
        # Track last seen frame number to purge aged tracks
        self.last_seen: Dict[int, int] = {}
        self.last_timestamp: Dict[int, float] = {}
        # Track class stability: tracking_id -> {class_name: hit_count}
        self.track_class_votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.max_lost_age = max_lost_age
        self.next_global_id = 1

    def update_tracks(self, detections: List[dict], frame_number: int, timestamp: float) -> List[dict]:
        # Step 0: Deduplicate incoming proposals in this frame (highest confidence / coffee first)
        sorted_detections = sorted(
            detections,
            key=lambda d: (1 if d.get("class_name") == "Gói Cà Phê" else 0, float(d.get("confidence", 0.0))),
            reverse=True,
        )
        accepted_detections: List[dict] = []
        for det in sorted_detections:
            b_det = (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))
            is_dup = False
            for acc in accepted_detections:
                b_acc = (float(acc["x1"]), float(acc["y1"]), float(acc["x2"]), float(acc["y2"]))
                if calculate_iou(b_det, b_acc) > 0.20 or calculate_center_distance(b_det, b_acc) < 45.0:
                    is_dup = True
                    break
            if not is_dup:
                accepted_detections.append(det)

        tracked: List[dict] = []
        matched_track_ids = set()

        # Step 1: Assign or match each detection against active tracks spatially
        for detection in accepted_detections:
            curr_box = (
                float(detection["x1"]),
                float(detection["y1"]),
                float(detection["x2"]),
                float(detection["y2"]),
            )
            raw_cls_name = detection["class_name"]

            # If tracking_id was already assigned externally
            tracking_id = detection.get("tracking_id")

            # Perform robust spatial matching across active tracks
            if tracking_id is None:
                best_match_id = None
                best_score = 0.0

                for cand_id, cand_obj in self.track_store.items():
                    if cand_id in matched_track_ids:
                        continue
                    cand_box = (cand_obj.x1, cand_obj.y1, cand_obj.x2, cand_obj.y2)
                    iou = calculate_iou(curr_box, cand_box)
                    dist = calculate_center_distance(curr_box, cand_box)

                    # Same class bonus: slightly prefer same class if multiple items are clustered
                    same_class_bonus = 0.15 if cand_obj.class_name == raw_cls_name else 0.0

                    # Match condition: IoU >= 0.15 OR centers are close (<= 95px)
                    if iou >= 0.15 or dist <= 95.0:
                        score = iou + max(0.0, 1.0 - (dist / 120.0)) + same_class_bonus
                        if score > best_score:
                            best_score = score
                            best_match_id = cand_id

                if best_match_id is not None:
                    tracking_id = best_match_id
                else:
                    # Guard: verify this new box does not heavily overlap with ANY already matched track
                    # to strictly prevent duplicate IDs on the same physical item
                    overlaps_existing = False
                    for existing_id in matched_track_ids:
                        existing_obj = self.track_store.get(existing_id)
                        if existing_obj:
                            ex_box = (existing_obj.x1, existing_obj.y1, existing_obj.x2, existing_obj.y2)
                            if calculate_iou(curr_box, ex_box) > 0.20 or calculate_center_distance(curr_box, ex_box) < 45.0:
                                overlaps_existing = True
                                break
                    if overlaps_existing:
                        continue  # Skip spawning duplicate track ID for the same object

                    tracking_id = self.next_global_id
                    self.next_global_id += 1
            else:
                self.next_global_id = max(self.next_global_id, tracking_id + 1)

            matched_track_ids.add(tracking_id)
            self.last_seen[tracking_id] = frame_number
            self.last_timestamp[tracking_id] = timestamp

            # Stabilize class name (prevent label flipping)
            self.track_class_votes[tracking_id][raw_cls_name] += 1
            stable_cls_name = max(self.track_class_votes[tracking_id].items(), key=lambda x: x[1])[0]

            tracked_item = {
                "tracking_id": tracking_id,
                "class_id": detection["class_id"],
                "class_name": stable_cls_name,
                "confidence": detection["confidence"],
                "x1": curr_box[0],
                "y1": curr_box[1],
                "x2": curr_box[2],
                "y2": curr_box[3],
                "frame_number": frame_number,
                "timestamp": timestamp,
            }
            tracked.append(tracked_item)

            # Update track store
            self.track_store[tracking_id] = TrackedObject(
                tracking_id=tracking_id,
                class_id=detection["class_id"],
                class_name=stable_cls_name,
                confidence=detection["confidence"],
                x1=curr_box[0],
                y1=curr_box[1],
                x2=curr_box[2],
                y2=curr_box[3],
                frame_number=frame_number,
                timestamp=timestamp,
            )

        # Step 2: Clean up tracks that have not been seen for too long
        for tid in list(self.track_store.keys()):
            if frame_number - self.last_seen.get(tid, 0) > self.max_lost_age:
                del self.track_store[tid]
                self.last_seen.pop(tid, None)
                self.last_timestamp.pop(tid, None)

        # Step 3: Merge duplicate overlapping tracks in store (guarantee at most 1 track per physical item)
        store_keys = list(self.track_store.keys())
        merged_ids = set()
        for i in range(len(store_keys)):
            for j in range(i + 1, len(store_keys)):
                id1, id2 = store_keys[i], store_keys[j]
                if id1 in merged_ids or id2 in merged_ids:
                    continue
                if id1 not in self.track_store or id2 not in self.track_store:
                    continue
                o1, o2 = self.track_store[id1], self.track_store[id2]
                b1 = (o1.x1, o1.y1, o1.x2, o1.y2)
                b2 = (o2.x1, o2.y1, o2.x2, o2.y2)
                if calculate_iou(b1, b2) > 0.25 or calculate_center_distance(b1, b2) < 45.0:
                    merged_ids.add(id2)
                    del self.track_store[id2]
                    self.last_seen.pop(id2, None)
                    self.last_timestamp.pop(id2, None)

        # Step 4: Prune any merged IDs from returned tracked detections
        final_tracked = [item for item in tracked if item["tracking_id"] in self.track_store and item["tracking_id"] not in merged_ids]
        return final_tracked

    def get_tracking_ids(self) -> Dict[str, set]:
        res = defaultdict(set)
        for tid, obj in self.track_store.items():
            res[obj.class_name].add(tid)
        return dict(res)

    def reset(self) -> None:
        self.track_store.clear()
        self.last_seen.clear()
        self.last_timestamp.clear()
        self.track_class_votes.clear()
        self.next_global_id = 1
