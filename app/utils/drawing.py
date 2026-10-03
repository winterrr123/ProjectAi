from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def draw_detection_overlay(frame: np.ndarray, detections: List[Dict[str, Any]], counters: Dict[str, int], total_count: int) -> np.ndarray:
    output_frame = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])
        is_counted = item.get("is_counted", False)
        status_text = "[COUNTED ✓]" if is_counted else "[NEW]"
        label = f"{item['class_name']} #{item.get('tracking_id', 'N/A')} {status_text} {item['confidence']:.0%}"
        # Emerald Green for counted, Electric Cyan for new
        color = (50, 220, 100) if is_counted else (240, 200, 0)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    y_offset = 30
    header = "PRODUCT DETECTION"
    cv2.putText(output_frame, header, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    for idx, item in enumerate(detections[:5]):
        text = f"ID: {item.get('tracking_id', 'N/A')} | {item['class_name']} | {item['confidence']:.0%}"
        cv2.putText(output_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    y_offset += 20
    cv2.putText(output_frame, f"Total: {total_count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 25
    for name, count in counters.items():
        cv2.putText(output_frame, f"{name}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 22
    return output_frame
