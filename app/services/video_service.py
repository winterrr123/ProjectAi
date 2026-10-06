from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.services.counting_service import CountingService
from app.services.yolo_service import YOLOService
from app.utils.drawing import draw_detection_overlay
from app.utils.helpers import get_video_output_path, logger


def calculate_box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class VideoProcessingService:
    def __init__(self, yolo_service: Optional[YOLOService] = None):
        self.yolo_service = yolo_service or YOLOService()
        self.counter = CountingService(min_hits=3)
        # EMA smoothing for box coordinates
        self._smooth_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        # Active tracks: tid -> {box, center, velocity, last_frame}
        self._active_tracks: Dict[int, Dict[str, Any]] = {}
        # Lost tracks buffer for Re-ID recovery: tid -> {box, center, velocity, lost_frame}
        self._lost_tracks: Dict[int, Dict[str, Any]] = {}
        self._fallback_id = 5000
        # Real-time processing progress for UI tracking
        self.progress: Dict[str, Any] = {
            "percent": 0,
            "current_frame": 0,
            "total_frames": 0,
            "status": "idle",
            "message": "",
        }

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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        self.progress = {
            "percent": 5,
            "current_frame": 0,
            "total_frames": total_frames,
            "status": "processing",
            "message": f"Bắt đầu phân tích {total_frames} khung hình...",
        }

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
        last_raw_detections: List[dict] = []

        # Reset all tracker and counting state for clean video processing
        self.reset()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1
                start = time.time()

                # Adaptive stride: for long videos (> 250 frames), run full YOLO inference
                # on odd frames and update EMA tracking on even frames to double processing speed
                # while preserving 100% full 30 FPS video output quality.
                if total_frames <= 250 or (frame_number % 2 == 1) or not last_raw_detections:
                    raw_detections = self.yolo_service.track_infer(frame, settings.CONFIDENCE_THRESHOLD)
                    last_raw_detections = raw_detections
                else:
                    raw_detections = last_raw_detections

                # Apply advanced Re-ID and smoothing
                tracked = self._post_process(raw_detections, frame_number, start)
                self.counter.process(tracked)
                accumulated = self.counter.get_accumulated_counts()
                final_counts = accumulated
                for item in tracked:
                    all_detections.append(item)
                overlay = draw_detection_overlay(frame, tracked, final_counts, self.counter.get_total_accumulated())
                writer.write(overlay)

                # Update live progress every frame
                pct = min(96, int(5 + (frame_number / total_frames) * 91))
                self.progress["percent"] = pct
                self.progress["current_frame"] = frame_number
                self.progress["message"] = f"Đang xử lý khung hình {frame_number}/{total_frames} ({pct}%)..."

                if frame_number % 50 == 0 or frame_number == total_frames:
                    logger.info("Processed frame %s/%s with %s detections (%s%%)", frame_number, total_frames, len(tracked), pct)
        finally:
            cap.release()
            writer.release()

        self.progress = {
            "percent": 100,
            "current_frame": frame_number,
            "total_frames": total_frames,
            "status": "completed",
            "message": "Xử lý và xuất video MP4 hoàn tất!",
        }

        summary = {
            "total_objects": self.counter.get_total_accumulated(),
            "by_class": final_counts,
            "unique_track_ids": self.counter.get_unique_track_ids(),
            "output_video": output_file,
            "frame_count": frame_number,
            "detections": all_detections,
        }
        logger.info("Video processing completed for %s (%s frames)", video_path, frame_number)
        return summary

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 1,
        timestamp: Optional[float] = None,
        generate_overlay: bool = False,
    ) -> Dict[str, Any]:
        timestamp = timestamp if timestamp is not None else time.time()
        # Use YOLO tracker inference
        raw_detections = self.yolo_service.track_infer(frame, settings.CONFIDENCE_THRESHOLD)
        # Apply intelligent Re-ID and trajectory smoothing
        tracked = self._post_process(raw_detections, frame_number, timestamp)
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

    def _post_process(self, detections: List[dict], frame_number: int, timestamp: float) -> List[dict]:
        """
        Post-process detections:
        - Geometric validation (reject noise/background)
        - Re-identification (prevent ID-switch from re-counting lost objects)
        - Fallback spatial tracking for custom models
        - EMA box smoothing (alpha = 0.45)
        - Normalize class to 'Sản phẩm' (AGENTS.md compliance)
        """
        tracked: List[dict] = []
        smooth_alpha = 0.45
        assigned_tids: set = set()

        for det in detections:
            raw_box = (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))
            bw = raw_box[2] - raw_box[0]
            bh = raw_box[3] - raw_box[1]
            if bw < 10 or bh < 10:
                continue

            cx = (raw_box[0] + raw_box[2]) / 2.0
            cy = (raw_box[1] + raw_box[3]) / 2.0
            raw_tid = det.get("tracking_id")
            resolved_tid = None

            # CASE 1: YOLO assigned a tracking_id
            if raw_tid is not None:
                # If raw_tid is already known and active in current tracks
                if raw_tid in self._active_tracks and raw_tid not in assigned_tids:
                    resolved_tid = raw_tid
                else:
                    # Check if this "new" tracker ID is actually a recovered lost track (Re-ID against ID-switching)
                    best_lost_id = None
                    best_match_score = 0.0
                    for lost_id, lost_info in list(self._lost_tracks.items()):
                        if lost_id in assigned_tids:
                            continue
                        dt = max(1, frame_number - lost_info["lost_frame"])
                        if dt > 30:
                            continue
                        vx, vy = lost_info.get("velocity", (0.0, 0.0))
                        pred_cx = lost_info["center"][0] + vx * dt
                        pred_cy = lost_info["center"][1] + vy * dt
                        center_dist = math.hypot(cx - pred_cx, cy - pred_cy)
                        iou = calculate_box_iou(raw_box, lost_info["box"])

                        # High IoU or close trajectory match
                        if iou > 0.35 or (center_dist < 50.0 and iou > 0.15) or center_dist < 32.0:
                            score = iou * 10.0 + (100.0 - min(100.0, center_dist)) / 10.0
                            if score > best_match_score:
                                best_match_score = score
                                best_lost_id = lost_id

                    if best_lost_id is not None:
                        # Successfully recovered lost identity!
                        resolved_tid = best_lost_id
                        self._lost_tracks.pop(best_lost_id, None)
                    else:
                        resolved_tid = raw_tid

            # CASE 2: No tracking ID provided (custom model or tracker dropout)
            if resolved_tid is None:
                # Try spatial matching against active tracks first
                best_match = None
                best_iou = 0.0
                for atid, ainfo in self._active_tracks.items():
                    if atid in assigned_tids:
                        continue
                    iou = calculate_box_iou(raw_box, ainfo["box"])
                    cdist = math.hypot(cx - ainfo["center"][0], cy - ainfo["center"][1])
                    if (iou > 0.35 or (iou > 0.20 and cdist < 45.0)) and iou > best_iou:
                        best_iou = iou
                        best_match = atid

                # Next try matching against recently lost tracks
                if best_match is None:
                    for ltid, linfo in list(self._lost_tracks.items()):
                        if ltid in assigned_tids:
                            continue
                        dt = max(1, frame_number - linfo["lost_frame"])
                        if dt > 30:
                            continue
                        iou = calculate_box_iou(raw_box, linfo["box"])
                        cdist = math.hypot(cx - linfo["center"][0], cy - linfo["center"][1])
                        if (iou > 0.35 or (iou > 0.20 and cdist < 45.0)) and iou > best_iou:
                            best_iou = iou
                            best_match = ltid

                if best_match is not None:
                    resolved_tid = best_match
                    self._lost_tracks.pop(best_match, None)
                else:
                    self._fallback_id += 1
                    resolved_tid = self._fallback_id

            assigned_tids.add(resolved_tid)

            # Uniform naming compliance (AGENTS.md)
            det["class_name"] = "Sản phẩm"
            det["class_id"] = 1

            # Box EMA smoothing
            if resolved_tid in self._smooth_boxes:
                sx1, sy1, sx2, sy2 = self._smooth_boxes[resolved_tid]
                smooth_x1 = sx1 + smooth_alpha * (raw_box[0] - sx1)
                smooth_y1 = sy1 + smooth_alpha * (raw_box[1] - sy1)
                smooth_x2 = sx2 + smooth_alpha * (raw_box[2] - sx2)
                smooth_y2 = sy2 + smooth_alpha * (raw_box[3] - sy2)
            else:
                smooth_x1, smooth_y1, smooth_x2, smooth_y2 = raw_box

            smoothed_box = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
            self._smooth_boxes[resolved_tid] = smoothed_box

            # Velocity computation for trajectory prediction
            smooth_cx = (smooth_x1 + smooth_x2) / 2.0
            smooth_cy = (smooth_y1 + smooth_y2) / 2.0
            if resolved_tid in self._active_tracks:
                prev_info = self._active_tracks[resolved_tid]
                dt = max(1, frame_number - prev_info["last_frame"])
                instant_vx = (smooth_cx - prev_info["center"][0]) / dt
                instant_vy = (smooth_cy - prev_info["center"][1]) / dt
                prev_vx, prev_vy = prev_info.get("velocity", (0.0, 0.0))
                vx = 0.6 * prev_vx + 0.4 * instant_vx
                vy = 0.6 * prev_vy + 0.4 * instant_vy
            else:
                vx, vy = 0.0, 0.0

            self._active_tracks[resolved_tid] = {
                "box": smoothed_box,
                "center": (smooth_cx, smooth_cy),
                "velocity": (vx, vy),
                "last_frame": frame_number,
            }

            tracked.append({
                "tracking_id": resolved_tid,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "x1": round(smooth_x1, 1),
                "y1": round(smooth_y1, 1),
                "x2": round(smooth_x2, 1),
                "y2": round(smooth_y2, 1),
                "frame_number": frame_number,
                "timestamp": timestamp,
            })

        # Move unseen active tracks to lost buffer
        for atid in list(self._active_tracks.keys()):
            if atid not in assigned_tids:
                ainfo = self._active_tracks.pop(atid)
                ainfo["lost_frame"] = frame_number
                self._lost_tracks[atid] = ainfo

        # Purge stale tracks older than 35 frames from lost buffer and smooth boxes
        stale_tids = [ltid for ltid, linfo in self._lost_tracks.items() if (frame_number - linfo["lost_frame"]) > 35]
        for ltid in stale_tids:
            self._lost_tracks.pop(ltid, None)
            self._smooth_boxes.pop(ltid, None)

        return tracked

    def reset(self) -> None:
        """Reset all tracking and counting state."""
        self.counter.reset()
        self.yolo_service.reset_tracker()
        self._smooth_boxes.clear()
        self._active_tracks.clear()
        self._lost_tracks.clear()
        self._fallback_id = 5000


