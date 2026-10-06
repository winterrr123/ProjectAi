from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def draw_detection_overlay(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    counters: Dict[str, int],
    total_count: int,
) -> np.ndarray:
    output_frame = frame.copy()
    h, w = output_frame.shape[:2]

    # Draw bounding boxes and clean labels
    for item in detections:
        x1, y1 = max(0, int(item["x1"])), max(0, int(item["y1"]))
        x2, y2 = min(w, int(item["x2"])), min(h, int(item["y2"]))
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        is_counted = item.get("is_counted", False)
        track_id = item.get("tracking_id", "N/A")

        # AGENTS.md: xanh dương khi đang track (240, 140, 40), xanh lá khi đã đếm (50, 205, 50) in BGR
        color = (50, 205, 50) if is_counted else (240, 140, 40)
        label = f"[x] San pham #{track_id}" if is_counted else f"San pham #{track_id}"

        # Bounding box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        # Label background pill
        font_scale = 0.5
        font_thick = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        lbl_y1 = max(0, y1 - txt_h - 8)
        lbl_y2 = y1
        lbl_x2 = min(w, x1 + txt_w + 10)

        cv2.rectangle(output_frame, (x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        cv2.putText(
            output_frame,
            label,
            (x1 + 5, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    # Minimalist unobtrusive HUD badge in top-left corner
    badge_text = f"Tong da dem: {total_count}"
    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    overlay = output_frame.copy()
    cv2.rectangle(overlay, (15, 15), (25 + bw, 30 + bh), (20, 25, 30), -1)
    cv2.addWeighted(overlay, 0.75, output_frame, 0.25, 0, output_frame)
    cv2.rectangle(output_frame, (15, 15), (25 + bw, 30 + bh), (50, 205, 50), 1)
    cv2.putText(
        output_frame,
        badge_text,
        (20, 22 + bh),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output_frame

