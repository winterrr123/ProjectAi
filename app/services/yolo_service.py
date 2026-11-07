from __future__ import annotations

import os
import threading
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
        self._base_coco_model: Optional[YOLO] = None
        self._custom_box_model: Optional[YOLO] = None
        self._custom_coffee_model: Optional[YOLO] = None
        self._custom_supermarket_model: Optional[YOLO] = None
        self._base_coco_attempted: bool = False
        self._custom_box_attempted: bool = False
        self._custom_coffee_attempted: bool = False
        self._custom_supermarket_attempted: bool = False
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo_infer")
        self._tracker_initialized = False
        self._load_model_if_available()

    def _warmup_model(self, model: Optional[YOLO]) -> None:
        """Warm up model on dummy frame so first real inference has zero cold-start delay."""
        if model is None:
            return
        try:
            device = self.get_device()
            dummy = np.zeros((360, 416, 3), dtype=np.uint8)
            with torch.inference_mode():
                model.predict(dummy, imgsz=416, device=device, verbose=False)
        except Exception as e:
            logger.warning("Model warmup warning: %s", e)

    def _load_model_if_available(self) -> None:
        """Load and warm up ONLY the primary YOLO model at initialization to conserve GPU VRAM."""
        if not os.path.exists(self.model_path):
            logger.warning("Model not found at %s. Detection will be unavailable until the model is added.", self.model_path)
            return

        logger.info("Loading primary YOLO model from %s", self.model_path)
        self.model = YOLO(self.model_path)
        self._warmup_model(self.model)
        logger.info("Primary YOLO model loaded and warmed up successfully")

    # --- Lazy Loading Helper Methods & Properties for Auxiliary Models ---

    def get_base_coco_model(self) -> Optional[YOLO]:
        """Lazy-load the universal base COCO model on first demand."""
        if not self._base_coco_attempted:
            with self._lock:
                if not self._base_coco_attempted:
                    self._base_coco_attempted = True
                    base_coco_candidates = [
                        settings.PROJECT_ROOT / "yolov8n.pt",
                        settings.PROJECT_ROOT / "models" / "best_backup.pt",
                    ]
                    for b_path in base_coco_candidates:
                        if b_path.exists():
                            try:
                                logger.info("Lazy-loading universal base COCO model from %s", b_path)
                                self._base_coco_model = YOLO(str(b_path))
                                self._warmup_model(self._base_coco_model)
                                break
                            except Exception as e:
                                logger.warning("Could not load base COCO model from %s: %s", b_path, e)
        return self._base_coco_model

    @property
    def base_coco_model(self) -> Optional[YOLO]:
        return self.get_base_coco_model()

    @base_coco_model.setter
    def base_coco_model(self, value: Optional[YOLO]) -> None:
        self._base_coco_model = value
        self._base_coco_attempted = True

    def get_custom_box_model(self) -> Optional[YOLO]:
        """Lazy-load the specialist box/carton model on first demand."""
        if not self._custom_box_attempted:
            with self._lock:
                if not self._custom_box_attempted:
                    self._custom_box_attempted = True
                    custom_weights_path = settings.PROJECT_ROOT / "runs" / "detect" / "train_box" / "weights" / "best.pt"
                    if custom_weights_path.exists():
                        try:
                            logger.info("Lazy-loading custom trained box model from %s", custom_weights_path)
                            self._custom_box_model = YOLO(str(custom_weights_path))
                            self._warmup_model(self._custom_box_model)
                        except Exception as e:
                            logger.warning("Could not load custom box model: %s", e)
        return self._custom_box_model

    @property
    def custom_box_model(self) -> Optional[YOLO]:
        return self.get_custom_box_model()

    @custom_box_model.setter
    def custom_box_model(self, value: Optional[YOLO]) -> None:
        self._custom_box_model = value
        self._custom_box_attempted = True

    def get_custom_coffee_model(self) -> Optional[YOLO]:
        """Lazy-load the specialist coffee bag model on first demand."""
        if not self._custom_coffee_attempted:
            with self._lock:
                if not self._custom_coffee_attempted:
                    self._custom_coffee_attempted = True
                    coffee_weights_candidates = [
                        settings.PROJECT_ROOT / "models" / "coffee_best.pt",
                        settings.PROJECT_ROOT / "runs" / "detect" / "train_coffee_v2" / "weights" / "best.pt",
                        settings.PROJECT_ROOT / "runs" / "detect" / "train_coffee" / "weights" / "best.pt",
                    ]
                    for c_path in coffee_weights_candidates:
                        if c_path.exists():
                            try:
                                logger.info("Lazy-loading custom coffee model from %s", c_path)
                                self._custom_coffee_model = YOLO(str(c_path))
                                self._warmup_model(self._custom_coffee_model)
                                break
                            except Exception as e:
                                logger.warning("Could not load coffee model from %s: %s", c_path, e)
        return self._custom_coffee_model

    @property
    def custom_coffee_model(self) -> Optional[YOLO]:
        return self.get_custom_coffee_model()

    @custom_coffee_model.setter
    def custom_coffee_model(self, value: Optional[YOLO]) -> None:
        self._custom_coffee_model = value
        self._custom_coffee_attempted = True

    def get_custom_supermarket_model(self) -> Optional[YOLO]:
        """Lazy-load the specialist supermarket drinks model on first demand."""
        if not self._custom_supermarket_attempted:
            with self._lock:
                if not self._custom_supermarket_attempted:
                    self._custom_supermarket_attempted = True
                    supermarket_path = settings.PROJECT_ROOT / "models" / "supermarket_lab605.pt"
                    if supermarket_path.exists():
                        try:
                            logger.info("Lazy-loading supermarket model from %s", supermarket_path)
                            self._custom_supermarket_model = YOLO(str(supermarket_path))
                            self._warmup_model(self._custom_supermarket_model)
                        except Exception as e:
                            logger.warning("Could not load supermarket model: %s", e)
        return self._custom_supermarket_model

    @property
    def custom_supermarket_model(self) -> Optional[YOLO]:
        return self.get_custom_supermarket_model()

    @custom_supermarket_model.setter
    def custom_supermarket_model(self, value: Optional[YOLO]) -> None:
        self._custom_supermarket_model = value
        self._custom_supermarket_attempted = True

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
                    "raw_name": raw_name.lower(),
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
        Intelligent deduplication & NMS across ensemble model outputs:
        - Prioritizes specialist trained models over generic base COCO detections.
        - Preserves distinct tracking IDs assigned by ByteTrack from the same model source (Track ID invariance).
        - Suppresses true duplicate boxes (same physical object across models) with elevated IoU > 0.55.
        - Uses scale-normalized center distance offsets (dx_rel, dy_rel) with center-guarded containment
          (containment > 0.75 and dx_rel < 0.35 and dy_rel < 0.35) and coincident-center jitter deduplication
          (IoU > 0.35 and dx_rel < 0.25 and dy_rel < 0.25) to preserve side-by-side and partially occluded cans.
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
            w_a = max(1.0, box_a[2] - box_a[0])
            h_a = max(1.0, box_a[3] - box_a[1])
            area_a = w_a * h_a
            ca_x, ca_y = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
            tid_a = det.get("tracking_id")
            src_a = det.get("model_source")

            is_duplicate = False
            for k in kept:
                box_b = (k["x1"], k["y1"], k["x2"], k["y2"])
                w_b = max(1.0, box_b[2] - box_b[0])
                h_b = max(1.0, box_b[3] - box_b[1])
                area_b = w_b * h_b
                cb_x, cb_y = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
                tid_b = k.get("tracking_id")
                src_b = k.get("model_source")

                # Track ID invariance: if both detections share the same model_source
                # and have distinct, valid tracking_ids, skip duplicate suppression between them
                # (ByteTrack already proved they are separate objects over temporal frames)
                if tid_a is not None and tid_b is not None and tid_a != tid_b and src_a == src_b:
                    continue

                ix1 = max(box_a[0], box_b[0])
                iy1 = max(box_a[1], box_b[1])
                ix2 = min(box_a[2], box_b[2])
                iy2 = min(box_a[3], box_b[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0

                min_area = min(area_a, area_b)
                containment = inter / min_area if min_area > 0 else 0.0

                max_w = max(w_a, w_b)
                max_h = max(h_a, h_b)
                min_w = min(w_a, w_b)
                min_h = min(h_a, h_b)

                # Normalized relative center distance offsets
                dx_rel = abs(ca_x - cb_x) / max_w
                dy_rel = abs(ca_y - cb_y) / max_h
                dx_rel_min = abs(ca_x - cb_x) / min_w
                dy_rel_min = abs(ca_y - cb_y) / min_h

                # 1. Definitive duplicate detection (same can detected across ensemble models)
                if iou > 0.55:
                    is_duplicate = True
                    break

                # 2. High containment (smaller box almost completely inside larger box)
                # Guarded by center alignment to prevent dropping adjacent occluded cans
                if containment > 0.75 and dx_rel < 0.35 and dy_rel < 0.35 and dx_rel_min < 0.35 and dy_rel_min < 0.35:
                    is_duplicate = True
                    break

                # 3. Moderate IoU with nearly coincident centers (model jitter duplicate)
                if iou > 0.35 and dx_rel < 0.25 and dy_rel < 0.25:
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
        models_to_reset = [self.model]
        if self._base_coco_model is not None:
            models_to_reset.append(self._base_coco_model)
        for m in models_to_reset:
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
