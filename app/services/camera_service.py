from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from app.config import settings
from app.services.counting_service import CountingService
from app.services.tracking_service import ByteTrackService
from app.services.yolo_service import YOLOService
from app.utils.drawing import draw_detection_overlay
from app.utils.helpers import logger


class CameraService:
    def __init__(self, yolo_service: Optional[YOLOService] = None):
        self.camera = None
        self.active = False
        self.tracker = ByteTrackService()
        self.counter = CountingService()
        self.model = yolo_service or YOLOService()
        self.yolo_service = self.model

    def reset(self) -> None:
        self.counter.reset()
        self.tracker = ByteTrackService()
        self.model.reset_tracker()

    def start(self, camera_index: int = 0) -> None:
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError("Cannot open camera")
        self.active = True
        logger.info("Camera started on index %s", camera_index)

    def stop(self) -> None:
        if self.camera is not None:
            self.camera.release()
        self.active = False
        logger.info("Camera stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.active or self.camera is None:
            return None
        ok, frame = self.camera.read()
        if not ok:
            logger.warning("Unable to read camera frame")
            return None
        return frame

    def process_frame(self, frame: np.ndarray, frame_number: int = 1) -> Dict[str, Any]:
        inference = self.model.infer(frame, settings.CONFIDENCE_THRESHOLD)
        tracked = self.tracker.update_tracks(inference, frame_number, time.time())
        new_counts = self.counter.process(tracked)
        accumulated_counts = self.counter.get_accumulated_counts()
        total = self.counter.get_total_accumulated()
        overlay = draw_detection_overlay(frame, tracked, accumulated_counts, total)
        return {
            "detections": tracked,
            "counts": accumulated_counts,
            "new_counts": new_counts,
            "total": total,
            "frame": overlay,
        }
