from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.services.counting_service import CountingService
from app.services.tracking_service import ByteTrackService
from app.services.yolo_service import YOLOService
from app.utils.drawing import draw_detection_overlay
from app.utils.helpers import get_video_output_path, logger


class VideoProcessingService:
    def __init__(self, yolo_service: Optional[YOLOService] = None):
        self.yolo_service = yolo_service or YOLOService()
        self.tracker = ByteTrackService()
        self.counter = CountingService()

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_file = output_path or get_video_output_path(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Try browser-friendly H264 codec first, fallback to mp4v
        writer = None
        for codec in ["avc1", "H264", "mp4v"]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                if test_writer.isOpened():
                    writer = test_writer
                    break
            except Exception:
                continue

        if writer is None or not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError("Unable to open video writer")

        frame_number = 0
        all_detections: List[dict] = []
        final_counts: Dict[str, int] = {}

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1
                start = time.time()
                detection_candidates = self.yolo_service.infer(frame, settings.CONFIDENCE_THRESHOLD)
                tracked = self.tracker.update_tracks(detection_candidates, frame_number, start)
                self.counter.process(tracked)
                accumulated = self.counter.get_accumulated_counts()
                final_counts = accumulated
                for item in tracked:
                    all_detections.append(item)
                overlay = draw_detection_overlay(frame, tracked, final_counts, self.counter.get_total_accumulated())
                writer.write(overlay)
                logger.info("Processed frame %s with %s detections", frame_number, len(tracked))
        finally:
            cap.release()
            writer.release()

        summary = {
            "total_objects": self.counter.get_total_accumulated(),
            "by_class": final_counts,
            "unique_track_ids": self.counter.get_unique_track_ids(),
            "output_video": output_file,
            "frame_count": frame_number,
            "detections": all_detections,
        }
        logger.info("Video processing completed for %s", video_path)
        return summary

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 1,
        timestamp: Optional[float] = None,
        generate_overlay: bool = False,
    ) -> Dict[str, Any]:
        timestamp = timestamp if timestamp is not None else time.time()
        detection_candidates = self.yolo_service.infer(frame, settings.CONFIDENCE_THRESHOLD)
        tracked = self.tracker.update_tracks(detection_candidates, frame_number, timestamp)
        new_counts = self.counter.process(tracked)
        accumulated_counts = self.counter.get_accumulated_counts()
        total = self.counter.get_total_accumulated()
        overlay = draw_detection_overlay(frame, tracked, accumulated_counts, total) if generate_overlay else None
        result = {
            "detections": tracked,
            "counts": accumulated_counts,
            "new_counts": new_counts,
            "total": total,
            "frame": overlay,
        }
        return result
