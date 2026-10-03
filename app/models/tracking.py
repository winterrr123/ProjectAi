from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackedObject:
    tracking_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_number: int
    timestamp: float


@dataclass
class DetectionSummary:
    total_objects: int
    by_class: Dict[str, int] = field(default_factory=dict)
    unique_track_ids: Dict[str, set] = field(default_factory=dict)
    average_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_objects": self.total_objects,
            "by_class": dict(self.by_class),
            "unique_track_ids": {k: sorted(v) for k, v in self.unique_track_ids.items()},
            "average_confidence": round(self.average_confidence, 4),
        }
