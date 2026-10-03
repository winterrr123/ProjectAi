from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import settings
from app.utils.helpers import logger

# Friendly Vietnamese class name translation for products
VIETNAMESE_PRODUCT_NAMES = {
    "gói cà phê": "Gói Cà Phê",
    "coffee_bag": "Gói Cà Phê",
    "coffee": "Gói Cà Phê",
    "pouch": "Túi / Gói Sản Phẩm",
    "bottle": "Chai Nước",
    "cup": "Ly / Cốc",
    "wine glass": "Ly Thủy Tinh",
    "bowl": "Bát / Tô",
    "cell phone": "Điện Thoại",
    "book": "Sách / Vở",
    "laptop": "Máy Tính Laptop",
    "mouse": "Chuột Máy Tính",
    "keyboard": "Bàn Phím",
    "remote": "Điều Khiển",
    "apple": "Quả Táo",
    "banana": "Quả Chuối",
    "orange": "Quả Cam",
    "sandwich": "Bánh Sandwich",
    "donut": "Bánh Donut",
    "cake": "Bánh Ngọt",
    "pizza": "Bánh Pizza",
    "hot dog": "Xúc Xích",
    "broccoli": "Bông Cải Xanh",
    "carrot": "Củ Cà Rốt",
    "backpack": "Balo",
    "handbag": "Túi Xách",
    "suitcase": "Vali / Hộp Lớn",
    "umbrella": "Ô / Dù",
    "tie": "Cà Vạt",
    "scissors": "Cây Kéo",
    "clock": "Đồng Hồ",
    "vase": "Bình Hoa",
    "box": "Thùng / Hộp Sản Phẩm",
    "refrigerator": "Tủ Lạnh",
    "tv": "Màn Hình TV",
    "microwave": "Lò Vi Sóng",
    "oven": "Lò Nướng",
    "toaster": "Máy Nướng Bánh",
    "sink": "Bồn Rửa",
    "teddy bear": "Gấu Bông",
    "hair drier": "Máy Sấy Tóc",
    "toothbrush": "Bàn Chải",
    "fork": "Nĩa",
    "knife": "Dao",
    "spoon": "Thìa / Muỗng",
    "cola": "Nước Ngọt Cola",
    "sports-drink": "Nước Thể Thao / Tăng Lực",
    "tea": "Trà Đóng Chai",
    "water": "Nước Suối",
}

# Classes to ignore when counting products (human beings, room furniture, background)
IGNORED_CLASSES = {
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "sink",
    "bench",
}


class YOLOService:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model: Optional[YOLO] = None
        self.custom_box_model: Optional[YOLO] = None
        self.custom_coffee_model: Optional[YOLO] = None
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="yolo_infer")
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning("Model not found at %s. Detection will be unavailable until the model is added.", self.model_path)
            return

        logger.info("Loading primary YOLO model from %s", self.model_path)
        self.model = YOLO(self.model_path)

        # Check for custom trained product box weights
        custom_weights_path = settings.PROJECT_ROOT / "runs" / "detect" / "train_box" / "weights" / "best.pt"
        if custom_weights_path.exists():
            try:
                logger.info("Loading custom trained box model from %s", custom_weights_path)
                self.custom_box_model = YOLO(str(custom_weights_path))
            except Exception as e:
                logger.warning("Could not load custom box model: %s", e)

        # Check for custom trained conveyor coffee bag weights
        coffee_weights_candidates = [
            settings.PROJECT_ROOT / "models" / "coffee_best.pt",
            settings.PROJECT_ROOT / "runs" / "detect" / "train_coffee_v2" / "weights" / "best.pt",
            settings.PROJECT_ROOT / "runs" / "detect" / "train_coffee" / "weights" / "best.pt",
        ]
        for c_path in coffee_weights_candidates:
            if c_path.exists():
                try:
                    logger.info("Loading custom coffee model from %s", c_path)
                    self.custom_coffee_model = YOLO(str(c_path))
                    break
                except Exception as e:
                    logger.warning("Could not load coffee model from %s: %s", c_path, e)

        # Warm up models on dummy frame so first frame has zero cold-start delay
        try:
            device = self.get_device()
            dummy = np.zeros((360, 416, 3), dtype=np.uint8)
            with torch.inference_mode():
                if self.model is not None:
                    self.model.predict(dummy, imgsz=416, device=device, verbose=False)
                if self.custom_box_model is not None:
                    self.custom_box_model.predict(dummy, imgsz=416, device=device, verbose=False)
                if self.custom_coffee_model is not None:
                    self.custom_coffee_model.predict(dummy, imgsz=416, device=device, verbose=False)
            logger.info("YOLO models warmed up successfully")
        except Exception as e:
            logger.warning("Model warmup warning: %s", e)

    def infer(self, frame: np.ndarray, confidence_threshold: float = None) -> List[Dict[str, Any]]:
        # Balanced threshold (0.25) eliminates jitter/noise while reliably detecting products
        threshold = confidence_threshold if confidence_threshold is not None else 0.25
        detections: List[Dict[str, Any]] = []
        device = self.get_device()

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        if self.model is None:
            return detections

        try:
            with torch.inference_mode():
                # Parallel inference if custom coffee or box models are also present
                f_coffee = None
                if self.custom_coffee_model is not None:
                    coffee_conf = min(0.20, threshold) if threshold is not None else 0.20
                    f_coffee = self.executor.submit(
                        self.custom_coffee_model.predict,
                        frame,
                        imgsz=416,
                        conf=coffee_conf,
                        iou=0.35,
                        device=device,
                        verbose=False,
                    )

                f_box = None
                if self.custom_box_model is not None:
                    f_box = self.executor.submit(
                        self.custom_box_model.predict,
                        frame,
                        imgsz=416,
                        conf=max(0.35, threshold),
                        device=device,
                        verbose=False,
                    )

                results = self.model.predict(
                    frame,
                    imgsz=416,
                    conf=max(0.35, threshold),
                    device=device,
                    verbose=False,
                )

                for result in results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for box in boxes:
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        raw_name = result.names.get(cls_id, f"Class_{cls_id}").lower()

                        if raw_name in IGNORED_CLASSES:
                            continue

                        xyxy = box.xyxy[0].cpu().numpy()
                        bx1, by1, bx2, by2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        bw = bx2 - bx1
                        bh = by2 - by1

                        # Eliminate false positives on full screen room background
                        if (bw * bh) > (frame_area * 0.75) or bw > (frame_w * 0.95) or bh > (frame_h * 0.95):
                            continue
                        if bw < 10 or bh < 10 or (bw * bh) < (frame_area * 0.001):
                            continue

                        translated_name = VIETNAMESE_PRODUCT_NAMES.get(raw_name, raw_name.title())

                        detections.append({
                            "tracking_id": None,
                            "class_id": cls_id,
                            "class_name": translated_name,
                            "confidence": round(conf, 2),
                            "x1": bx1,
                            "y1": by1,
                            "x2": bx2,
                            "y2": by2,
                        })

                # Merge custom coffee bag detections
                if f_coffee is not None:
                    coffee_results = f_coffee.result()
                    for result in coffee_results:
                        boxes = result.boxes
                        if boxes is None or len(boxes) == 0:
                            continue
                        for box in boxes:
                            conf = float(box.conf[0].cpu().numpy())
                            xyxy = box.xyxy[0].cpu().numpy()
                            bx1, by1, bx2, by2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                            bw = bx2 - bx1
                            bh = by2 - by1

                            if (bw * bh) > (frame_area * 0.75) or bw > (frame_w * 0.95) or bh > (frame_h * 0.95):
                                continue
                            if bw < 10 or bh < 10 or (bw * bh) < (frame_area * 0.001):
                                continue

                            # Check overlap with existing detections (IoU > 0.35)
                            is_dup = False
                            for d in detections:
                                dx1, dy1, dx2, dy2 = d["x1"], d["y1"], d["x2"], d["y2"]
                                inter = max(0.0, min(bx2, dx2) - max(bx1, dx1)) * max(0.0, min(by2, dy2) - max(by1, dy1))
                                union = (bx2 - bx1) * (by2 - by1) + (dx2 - dx1) * (dy2 - dy1) - inter
                                if union > 0 and (inter / union) > 0.35:
                                    # Overwrite generic label with accurate coffee bag label
                                    d["class_name"] = "Gói Cà Phê"
                                    d["class_id"] = 100
                                    d["confidence"] = max(d["confidence"], round(conf, 2))
                                    is_dup = True
                                    break

                            if not is_dup:
                                detections.append({
                                    "tracking_id": None,
                                    "class_id": 100,
                                    "class_name": "Gói Cà Phê",
                                    "confidence": round(conf, 2),
                                    "x1": bx1,
                                    "y1": by1,
                                    "x2": bx2,
                                    "y2": by2,
                                })

                # Merge custom box detections (if available)
                if f_box is not None:
                    box_results = f_box.result()
                    for result in box_results:
                        boxes = result.boxes
                        if boxes is None or len(boxes) == 0:
                            continue
                        for box in boxes:
                            conf = float(box.conf[0].cpu().numpy())
                            xyxy = box.xyxy[0].cpu().numpy()
                            bx1, by1, bx2, by2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                            bw = bx2 - bx1
                            bh = by2 - by1

                            if (bw * bh) > (frame_area * 0.75) or bw > (frame_w * 0.95) or bh > (frame_h * 0.95):
                                continue
                            if bw < 10 or bh < 10:
                                continue

                            # Check overlap with existing detections to avoid duplicates (IoU > 0.4)
                            is_dup = False
                            for d in detections:
                                dx1, dy1, dx2, dy2 = d["x1"], d["y1"], d["x2"], d["y2"]
                                inter = max(0.0, min(bx2, dx2) - max(bx1, dx1)) * max(0.0, min(by2, dy2) - max(by1, dy1))
                                union = (bx2 - bx1) * (by2 - by1) + (dx2 - dx1) * (dy2 - dy1) - inter
                                if union > 0 and (inter / union) > 0.4:
                                    is_dup = True
                                    break

                            if not is_dup:
                                detections.append({
                                    "tracking_id": None,
                                    "class_id": 99,
                                    "class_name": "Thùng / Hộp Sản Phẩm",
                                    "confidence": round(conf, 2),
                                    "x1": bx1,
                                    "y1": by1,
                                    "x2": bx2,
                                    "y2": by2,
                                })
        except Exception as e:
            logger.error("YOLO inference error: %s", e)

        return self.suppress_overlapping_boxes(detections)

    @staticmethod
    def suppress_overlapping_boxes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Strict deduplication & NMS across all model outputs:
        - If two boxes have IoU > 0.25, keep only the higher-confidence one.
        - If one box is mostly contained inside another (containment > 0.45), keep only one.
        - If centers are within 40px and have overlap, merge them.
        - Strictly prevents multiple boxes on the same physical object.
        """
        if len(detections) <= 1:
            return detections

        def score(d):
            is_coffee = 1.0 if d.get("class_name") == "Gói Cà Phê" else 0.0
            return is_coffee * 10.0 + float(d.get("confidence", 0.0))

        sorted_dets = sorted(detections, key=score, reverse=True)
        kept: List[Dict[str, Any]] = []

        for det in sorted_dets:
            box_a = (det["x1"], det["y1"], det["x2"], det["y2"])
            area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))

            is_duplicate = False
            for k in kept:
                box_b = (k["x1"], k["y1"], k["x2"], k["y2"])
                area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

                ix1 = max(box_a[0], box_b[0])
                iy1 = max(box_a[1], box_b[1])
                ix2 = min(box_a[2], box_b[2])
                iy2 = min(box_a[3], box_b[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0

                min_area = min(area_a, area_b)
                containment = inter / min_area if min_area > 0 else 0.0

                ca_x, ca_y = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
                cb_x, cb_y = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
                dist = ((ca_x - cb_x) ** 2 + (ca_y - cb_y) ** 2) ** 0.5

                if iou > 0.25 or containment > 0.45 or (dist < 40.0 and iou > 0.10):
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(det)

        return kept

    @staticmethod
    def get_device() -> str:
        raw_device = str(settings.DEVICE).strip().lower()
        if torch.cuda.is_available():
            if raw_device.startswith("cuda"):
                return raw_device
            return "cuda"

        if raw_device == "cpu":
            return "cpu"

        return "cpu"
