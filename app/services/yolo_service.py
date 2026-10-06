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
    # Động vật / Thú cưng
    "dog": "Con Chó",
    "cat": "Con Mèo",
    "bird": "Con Chim",
    "horse": "Con Ngựa",
    "sheep": "Con Cừu",
    "cow": "Con Bò",
    "elephant": "Con Voi",
    "bear": "Con Gấu",
    "zebra": "Ngựa Vằn",
    "giraffe": "Hươu Cao Cổ",
    "fish": "Con Cá",
    "con cá": "Con Cá",
    "cá": "Con Cá",
    "người": "Người",
    "sản phẩm": "Sản phẩm",
}

# Classes to ignore when counting products (human beings, room furniture, background)
IGNORED_CLASSES = {
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "sink",
    "bench",
    "kite",
    "airplane",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
}


class YOLOService:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model: Optional[YOLO] = None
        self.base_coco_model: Optional[YOLO] = None
        self.custom_box_model: Optional[YOLO] = None
        self.custom_coffee_model: Optional[YOLO] = None
        self.custom_supermarket_model: Optional[YOLO] = None
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="yolo_infer")
        self._tracker_initialized = False
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning("Model not found at %s. Detection will be unavailable until the model is added.", self.model_path)
            return

        logger.info("Loading primary YOLO model from %s", self.model_path)
        self.model = YOLO(self.model_path)

        # 1. Base Universal COCO Model (80 classes: orange, bottle, cup, laptop, etc.)
        # Provides universal general object recognition alongside specialist trained models
        base_coco_candidates = [
            settings.PROJECT_ROOT / "yolov8n.pt",
            settings.PROJECT_ROOT / "models" / "best_backup.pt",
        ]
        for b_path in base_coco_candidates:
            if b_path.exists():
                try:
                    logger.info("Loading universal base COCO model from %s", b_path)
                    self.base_coco_model = YOLO(str(b_path))
                    break
                except Exception as e:
                    logger.warning("Could not load base COCO model from %s: %s", b_path, e)

        # 2. Specialist product box model
        custom_weights_path = settings.PROJECT_ROOT / "runs" / "detect" / "train_box" / "weights" / "best.pt"
        if custom_weights_path.exists():
            try:
                logger.info("Loading custom trained box model from %s", custom_weights_path)
                self.custom_box_model = YOLO(str(custom_weights_path))
            except Exception as e:
                logger.warning("Could not load custom box model: %s", e)

        # 3. Specialist conveyor coffee bag model
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

        # 4. Specialist supermarket drinks model
        supermarket_path = settings.PROJECT_ROOT / "models" / "supermarket_lab605.pt"
        if supermarket_path.exists():
            try:
                logger.info("Loading supermarket model from %s", supermarket_path)
                self.custom_supermarket_model = YOLO(str(supermarket_path))
            except Exception as e:
                logger.warning("Could not load supermarket model: %s", e)

        # Warm up models on dummy frame so first frame has zero cold-start delay
        try:
            device = self.get_device()
            dummy = np.zeros((360, 416, 3), dtype=np.uint8)
            with torch.inference_mode():
                if self.model is not None:
                    self.model.predict(dummy, imgsz=416, device=device, verbose=False)
                if self.base_coco_model is not None:
                    self.base_coco_model.predict(dummy, imgsz=416, device=device, verbose=False)
                if self.custom_box_model is not None:
                    self.custom_box_model.predict(dummy, imgsz=416, device=device, verbose=False)
                if self.custom_coffee_model is not None:
                    self.custom_coffee_model.predict(dummy, imgsz=416, device=device, verbose=False)
                if self.custom_supermarket_model is not None:
                    self.custom_supermarket_model.predict(dummy, imgsz=416, device=device, verbose=False)
            logger.info("YOLO multi-model ensemble warmed up successfully")
        except Exception as e:
            logger.warning("Model warmup warning: %s", e)

    def _extract_boxes(
        self,
        results: Any,
        model_source: str,
        id_offset: int = 0,
        forced_class_id: Optional[int] = None,
        forced_class_name: Optional[str] = None,
        frame_area: float = 0.0,
        frame_w: float = 0.0,
        frame_h: float = 0.0,
    ) -> List[Dict[str, Any]]:
        extracted: List[Dict[str, Any]] = []
        if results is None:
            return extracted

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                raw_name = result.names.get(cls_id, f"Class_{cls_id}").lower() if hasattr(result, "names") else f"Class_{cls_id}"

                if forced_class_name is None and raw_name in IGNORED_CLASSES:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                bx1, by1, bx2, by2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                bw = bx2 - bx1
                bh = by2 - by1

                # Eliminate false positives on full screen room background
                max_ratio = 0.98 if raw_name in ("người", "person") else 0.85
                if frame_area > 0 and ((bw * bh) > (frame_area * max_ratio) or (bw > (frame_w * 0.98) and bh > (frame_h * 0.98))):
                    continue
                if bw < 8 or bh < 8 or (frame_area > 0 and (bw * bh) < (frame_area * 0.0003)):
                    continue

                track_id = None
                if getattr(box, "id", None) is not None:
                    track_id = int(box.id[0].cpu().numpy()) + id_offset

                final_cls_id = forced_class_id if forced_class_id is not None else cls_id
                final_name = forced_class_name if forced_class_name is not None else VIETNAMESE_PRODUCT_NAMES.get(raw_name, raw_name.title())

                extracted.append({
                    "tracking_id": track_id,
                    "class_id": final_cls_id,
                    "class_name": final_name,
                    "confidence": round(conf, 2),
                    "x1": bx1,
                    "y1": by1,
                    "x2": bx2,
                    "y2": by2,
                    "model_source": model_source,
                })

        return extracted

    @staticmethod
    def suppress_overlapping_boxes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Strict deduplication & NMS across all model outputs in the ensemble:
        - Prioritizes specialist trained models (Người, Con Cá, Gói Cà Phê, Thùng Hộp, Nước Ngọt, Trà...) over generic base COCO detections.
        - If two boxes have IoU > 0.25, keep only the higher-priority / higher-confidence one.
        - If one box is mostly contained inside another (containment > 0.45), keep only one.
        - If centers are within 40px and have overlap, merge them.
        - Strictly prevents duplicate bounding boxes on the same physical object.
        """
        if len(detections) <= 1:
            return detections

        def score(d: Dict[str, Any]) -> float:
            src = d.get("model_source", "")
            cls = d.get("class_name", "")

            # Specialist models get strong priority when competing with generic base detections
            if src in ("specialist", "custom_coffee", "supermarket", "custom_box"):
                source_priority = 20.0
            elif cls in ("Con Cá", "Gói Cà Phê", "Thùng / Hộp Sản Phẩm", "Nước Ngọt Cola", "Trà Đóng Chai", "Nước Suối", "Nước Thể Thao / Tăng Lực", "Người"):
                source_priority = 15.0
            else:
                source_priority = 0.0

            # Prioritize detections that already have an established tracking_id
            has_track = 5.0 if d.get("tracking_id") is not None else 0.0

            return source_priority + has_track + float(d.get("confidence", 0.0))

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

    def infer(self, frame: np.ndarray, confidence_threshold: float = None) -> List[Dict[str, Any]]:
        threshold = confidence_threshold if confidence_threshold is not None else 0.25
        detections: List[Dict[str, Any]] = []
        device = self.get_device()

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        # Check active vision engine from VisionManager
        try:
            from app.services.vision_api_service import vision_manager
            if vision_manager.active_engine == "yolo_world" and vision_manager.yolo_world_service and vision_manager.yolo_world_service.is_loaded:
                world_dets = vision_manager.yolo_world_service.infer(frame, confidence_threshold=threshold)
                return self.suppress_overlapping_boxes(world_dets)
            elif vision_manager.active_engine == "hybrid" and vision_manager.yolo_world_service and vision_manager.yolo_world_service.is_loaded:
                world_dets = vision_manager.yolo_world_service.infer(frame, confidence_threshold=threshold)
                detections.extend(world_dets)
        except Exception as e:
            logger.debug("Vision engine dispatch note: %s", e)

        if self.model is None and self.base_coco_model is None and not detections:
            return detections

        try:
            with torch.inference_mode():
                futures = {}

                # 1. Primary specialist model (Sản Phẩm, Người, Con Cá)
                if self.model is not None:
                    spec_conf = min(0.20, threshold) if threshold is not None else 0.20
                    futures["specialist"] = self.executor.submit(
                        self.model.predict,
                        frame,
                        imgsz=416,
                        conf=spec_conf,
                        device=device,
                        verbose=False,
                    )

                # 2. Universal base COCO model (80 classes: orange, bottle, cup, laptop, etc.)
                if self.base_coco_model is not None:
                    futures["base"] = self.executor.submit(
                        self.base_coco_model.predict,
                        frame,
                        imgsz=416,
                        conf=max(0.25, threshold),
                        device=device,
                        verbose=False,
                    )

                # 3. Supermarket drinks specialist model
                if self.custom_supermarket_model is not None:
                    futures["supermarket"] = self.executor.submit(
                        self.custom_supermarket_model.predict,
                        frame,
                        imgsz=416,
                        conf=max(0.25, threshold),
                        device=device,
                        verbose=False,
                    )

                # 4. Conveyor coffee bag specialist model
                if self.custom_coffee_model is not None:
                    coffee_conf = min(0.20, threshold) if threshold is not None else 0.20
                    futures["coffee"] = self.executor.submit(
                        self.custom_coffee_model.predict,
                        frame,
                        imgsz=416,
                        conf=coffee_conf,
                        iou=0.35,
                        device=device,
                        verbose=False,
                    )

                # 5. Box / Carton specialist model
                if self.custom_box_model is not None:
                    futures["box"] = self.executor.submit(
                        self.custom_box_model.predict,
                        frame,
                        imgsz=416,
                        conf=max(0.35, threshold),
                        device=device,
                        verbose=False,
                    )

                # Gather and extract detections from all futures
                if "specialist" in futures:
                    res = futures["specialist"].result()
                    detections.extend(self._extract_boxes(res, model_source="specialist", frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

                if "base" in futures:
                    res = futures["base"].result()
                    detections.extend(self._extract_boxes(res, model_source="base", frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

                if "supermarket" in futures:
                    res = futures["supermarket"].result()
                    detections.extend(self._extract_boxes(res, model_source="supermarket", frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

                if "coffee" in futures:
                    res = futures["coffee"].result()
                    detections.extend(self._extract_boxes(res, model_source="custom_coffee", forced_class_id=100, forced_class_name="Gói Cà Phê", frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

                if "box" in futures:
                    res = futures["box"].result()
                    detections.extend(self._extract_boxes(res, model_source="custom_box", forced_class_id=99, forced_class_name="Thùng / Hộp Sản Phẩm", frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

        except Exception as e:
            logger.error("YOLO ensemble inference error: %s", e)

        return self.suppress_overlapping_boxes(detections)

    def track_infer(self, frame: np.ndarray, confidence_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Multi-model ensemble tracking:
        - Primary specialist model (.track with ByteTrack for custom trained classes: Con Cá, Người, Sản Phẩm)
        - Base COCO model (.track with ByteTrack for universal objects: orange, bottle, cup, laptop, etc. with ID offset +10000)
        - Auxiliary specialist models (.predict for coffee, drinks, boxes)
        - Intelligent suppress_overlapping_boxes deduplication prioritizing specialist detections
        """
        threshold = confidence_threshold if confidence_threshold is not None else 0.25
        detections: List[Dict[str, Any]] = []
        device = self.get_device()

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        # Check active vision engine from VisionManager
        try:
            from app.services.vision_api_service import vision_manager
            if vision_manager.active_engine == "yolo_world" and vision_manager.yolo_world_service and vision_manager.yolo_world_service.is_loaded:
                world_dets = vision_manager.yolo_world_service.infer(frame, confidence_threshold=threshold)
                return self.suppress_overlapping_boxes(world_dets)
            elif vision_manager.active_engine == "hybrid" and vision_manager.yolo_world_service and vision_manager.yolo_world_service.is_loaded:
                world_dets = vision_manager.yolo_world_service.infer(frame, confidence_threshold=threshold)
                detections.extend(world_dets)
        except Exception as e:
            logger.debug("Vision engine dispatch note: %s", e)

        if self.model is None and self.base_coco_model is None and not detections:
            return detections

        try:
            with torch.inference_mode():
                futures = {}

                # 1. Primary specialist model with ByteTrack
                if self.model is not None:
                    spec_conf = min(0.20, threshold) if threshold is not None else 0.20
                    futures["specialist"] = self.executor.submit(
                        self.model.track,
                        frame,
                        imgsz=416,
                        conf=spec_conf,
                        device=device,
                        verbose=False,
                        persist=True,
                        tracker="bytetrack.yaml",
                    )
                    self._tracker_initialized = True

                # 2. Universal base COCO model with ByteTrack (offset ID by +10000 to prevent collisions)
                if self.base_coco_model is not None:
                    futures["base"] = self.executor.submit(
                        self.base_coco_model.track,
                        frame,
                        imgsz=416,
                        conf=max(0.25, threshold),
                        device=device,
                        verbose=False,
                        persist=True,
                        tracker="bytetrack.yaml",
                    )
                    self._tracker_initialized = True

                # 3. Supermarket drinks specialist model
                if self.custom_supermarket_model is not None:
                    futures["supermarket"] = self.executor.submit(
                        self.custom_supermarket_model.predict,
                        frame,
                        imgsz=416,
                        conf=max(0.25, threshold),
                        device=device,
                        verbose=False,
                    )

                # 4. Conveyor coffee bag specialist model
                if self.custom_coffee_model is not None:
                    coffee_conf = min(0.20, threshold) if threshold is not None else 0.20
                    futures["coffee"] = self.executor.submit(
                        self.custom_coffee_model.predict,
                        frame,
                        imgsz=416,
                        conf=coffee_conf,
                        iou=0.35,
                        device=device,
                        verbose=False,
                    )

                # 5. Box / Carton specialist model
                if self.custom_box_model is not None:
                    futures["box"] = self.executor.submit(
                        self.custom_box_model.predict,
                        frame,
                        imgsz=416,
                        conf=max(0.35, threshold),
                        device=device,
                        verbose=False,
                    )

                # Gather and extract detections from all futures
                if "specialist" in futures:
                    res = futures["specialist"].result()
                    detections.extend(self._extract_boxes(res, model_source="specialist", id_offset=0, frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

                if "base" in futures:
                    res = futures["base"].result()
                    detections.extend(self._extract_boxes(res, model_source="base", id_offset=10000, frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

                if "supermarket" in futures:
                    res = futures["supermarket"].result()
                    detections.extend(self._extract_boxes(res, model_source="supermarket", frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

                if "coffee" in futures:
                    res = futures["coffee"].result()
                    detections.extend(self._extract_boxes(res, model_source="custom_coffee", forced_class_id=100, forced_class_name="Gói Cà Phê", frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

                if "box" in futures:
                    res = futures["box"].result()
                    detections.extend(self._extract_boxes(res, model_source="custom_box", forced_class_id=99, forced_class_name="Thùng / Hộp Sản Phẩm", frame_area=frame_area, frame_w=frame_w, frame_h=frame_h))

        except Exception as e:
            logger.error("YOLO track ensemble error: %s", e)

        return self.suppress_overlapping_boxes(detections)

    def reset_tracker(self) -> None:
        """Reset the YOLO built-in tracker state for all tracking models."""
        for m in (self.model, self.base_coco_model):
            if m is not None:
                try:
                    if hasattr(m, "predictor") and m.predictor is not None:
                        if hasattr(m.predictor, "trackers"):
                            for tracker in m.predictor.trackers:
                                tracker.reset()
                        m.predictor.trackers = []
                except Exception as e:
                    logger.warning("Tracker reset note: %s", e)
        self._tracker_initialized = False
        logger.info("YOLO multi-model trackers reset successfully")

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
