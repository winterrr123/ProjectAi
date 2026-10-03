"""
==============================================================================
PRODUCT VISION - OPEN-VOCABULARY & MULTIMODAL VISION AI SERVICE
Hỗ trợ nhận diện vạn vật qua:
1. Google Gemini 1.5 / 2.0 Flash Vision (REST API - Brand & Label Recognition)
2. Ultralytics YOLO-World v2 (Open-Vocabulary Zero-Shot Object Detection)
3. Caching & Hybrid Tracking Pipeline
==============================================================================
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from app.config import settings
from app.utils.helpers import logger


class VisionCache:
    """In-memory LRU cache with SHA256 image hashes to avoid repeated API charges."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 86400):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            ts, val = self.cache[key]
            if time.time() - ts < self.ttl:
                return val
            del self.cache[key]
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if len(self.cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
        self.cache[key] = (time.time(), value)

    def hash_image(self, img_bytes: bytes, salt: str = "") -> str:
        h = hashlib.sha256(img_bytes)
        if salt:
            h.update(salt.encode("utf-8"))
        return h.hexdigest()


class GeminiVisionService:
    """
    Google Gemini 1.5 Flash / 2.0 Flash REST Client.
    Direct HTTP calls via requests without heavy external SDK overhead.
    """

    DEFAULT_MODEL = "gemini-1.5-flash"
    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.cache = VisionCache()

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    def get_api_key(self) -> str:
        return self.api_key or settings.GEMINI_API_KEY or ""

    def test_key(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Verify API key validity against Gemini REST endpoint."""
        key = api_key or self.get_api_key()
        if not key:
            return {"valid": False, "message": "Chưa cung cấp Gemini API Key."}

        url = self.API_URL_TEMPLATE.format(model=self.DEFAULT_MODEL, key=key)
        payload = {
            "contents": [
                {
                    "parts": [{"text": "Hello, respond with the single word 'OK'."}]
                }
            ],
            "generationConfig": {"maxOutputTokens": 10, "temperature": 0.0},
        }

        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                return {"valid": True, "message": "Gemini API Key hợp lệ và sẵn sàng hoạt động!"}
            err_data = resp.json().get("error", {})
            return {
                "valid": False,
                "message": err_data.get("message", f"Mã lỗi HTTP {resp.status_code}"),
            }
        except Exception as e:
            return {"valid": False, "message": f"Lỗi kết nối tới Gemini API: {str(e)}"}

    def identify_product(
        self,
        image_bytes: bytes,
        custom_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deep product recognition on image or cropped object box:
        Reads brand, product name, weight, condition, barcode, and packaging details in Vietnamese.
        """
        key = api_key or self.get_api_key()
        if not key:
            return {
                "success": False,
                "error": "Chưa cấu hình Gemini API Key. Vui lòng nhập API Key trong bảng điều khiển.",
            }

        cache_key = self.cache.hash_image(image_bytes, salt="identify")
        cached = self.cache.get(cache_key)
        if cached:
            cached["cached"] = True
            return cached

        # Prepare base64
        b64_img = base64.b64encode(image_bytes).decode("utf-8")

        prompt = custom_prompt or (
            "Bạn là chuyên gia thị giác máy tính trong hệ thống nhận diện sản phẩm bán lẻ và công nghiệp. "
            "Hãy kiểm tra và đọc kỹ nhãn hiệu, bao bì, chữ viết và logo trên hình ảnh sản phẩm này. "
            "Trả về DUY NHẤT một chuỗi JSON chuẩn (không bọc trong markdown codeblock ```json) "
            "với các trường chính xác sau:\n"
            "{\n"
            '  "product_name": "Tên chi tiết của sản phẩm (tiếng Việt, ví dụ: Cà Phê G7 3in1 / Nước Tăng Lực Red Bull)",\n'
            '  "brand": "Tên thương hiệu (ví dụ: Trung Nguyên / Coca-Cola / Vinamilk / Chinsu / Unicharm / Hảo Hảo)",\n'
            '  "category": "Ngành hàng (ví dụ: Cà phê hòa tan / Nước giải khát / Bánh kẹo / Sữa tươi / Gia vị)",\n'
            '  "description": "Mô tả ngắn gọn về kiểu dáng, màu sắc và bao bì sản phẩm (1-2 câu)",\n'
            '  "specs": "Quy cách đóng gói hoặc trọng lượng/dung tích nếu đọc được (ví dụ: 16g / 330ml / 500g)",\n'
            '  "packaging_condition": "Tình trạng bao bì (ví dụ: Nguyên vẹn, mới / Hơi nhăn / Biến dạng)",\n'
            '  "barcode_or_sku": "Mã vạch hoặc số hiệu SKU nếu nhìn thấy trên bao bì, nếu không ghi None",\n'
            '  "confidence": 0.95\n'
            "}"
        )

        url = self.API_URL_TEMPLATE.format(model=self.DEFAULT_MODEL, key=key)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_img,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 32,
                "topP": 0.9,
                "maxOutputTokens": 800,
                "responseMimeType": "application/json",
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
                return {"success": False, "error": f"Lỗi Gemini Vision API: {err_msg}"}

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return {"success": False, "error": "Gemini không trả về kết quả nhận diện."}

            text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

            # Clean JSON markdown fences if any
            clean_json = re.sub(r"^```json\s*", "", text_content)
            clean_json = re.sub(r"^```\s*", "", clean_json)
            clean_json = re.sub(r"\s*```$", "", clean_json)

            try:
                parsed = json.loads(clean_json)
            except Exception:
                # Fallback parser using regex or raw text
                parsed = {
                    "product_name": "Sản Phẩm Đã Nhận Diện",
                    "brand": "Chưa rõ",
                    "category": "Hàng tiêu dùng",
                    "description": text_content[:200],
                    "specs": "Tiêu chuẩn",
                    "packaging_condition": "Nguyên vẹn",
                    "confidence": 0.85,
                }

            result = {
                "success": True,
                "cached": False,
                "data": parsed,
                "raw_text": text_content,
            }
            self.cache.set(cache_key, result)
            return result

        except requests.exceptions.Timeout:
            return {"success": False, "error": "Kết nối Gemini API bị quá thời gian (Timeout)."}
        except Exception as e:
            logger.error("Gemini Vision API error: %s", e)
            return {"success": False, "error": f"Lỗi xử lý Gemini: {str(e)}"}

    def analyze_full_scene(
        self,
        image_bytes: bytes,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multimodal scene analysis: counts all items, lists all brands, and provides summary.
        """
        key = api_key or self.get_api_key()
        if not key:
            return {
                "success": False,
                "error": "Chưa cấu hình Gemini API Key.",
            }

        cache_key = self.cache.hash_image(image_bytes, salt="scene")
        cached = self.cache.get(cache_key)
        if cached:
            cached["cached"] = True
            return cached

        b64_img = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "Phân tích toàn cảnh hình ảnh/khung hình này cho hệ thống Computer Vision đếm sản phẩm. "
            "Trả về chuỗi JSON thuần túy (không bọc trong markdown codeblock ```json) cấu trúc:\n"
            "{\n"
            '  "scene_summary": "Tóm tắt cảnh quay (băng chuyền, bàn làm việc, kệ hàng...)",\n'
            '  "total_items": 5,\n'
            '  "detected_products": [\n'
            '     {"name": "Gói Cà Phê G7", "brand": "Trung Nguyên", "count": 3, "location": "giữa khung hình"},\n'
            '     {"name": "Chai Nước Suối", "brand": "Aquafina", "count": 2, "location": "bên phải"}\n'
            "  ],\n"
            '  "inspection_notes": "Nhận xét về mật độ, độ sáng và chất lượng hình ảnh"\n'
            "}"
        )

        url = self.API_URL_TEMPLATE.format(model=self.DEFAULT_MODEL, key=key)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_img,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code != 200:
                err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
                return {"success": False, "error": f"Lỗi Gemini: {err_msg}"}

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return {"success": False, "error": "Không có phản hồi từ Gemini."}

            text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            clean_json = re.sub(r"^```json\s*", "", text_content)
            clean_json = re.sub(r"^```\s*", "", clean_json)
            clean_json = re.sub(r"\s*```$", "", clean_json)

            try:
                parsed = json.loads(clean_json)
            except Exception:
                parsed = {"scene_summary": text_content, "total_items": 0, "detected_products": []}

            result = {"success": True, "cached": False, "data": parsed}
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


class YOLOWorldService:
    """
    Open-Vocabulary Zero-Shot Object Detection using Ultralytics YOLO-World v2.
    Allows specifying ANY arbitrary Vietnamese or English product prompt on the fly!
    """

    MODEL_NAME = "yolov8s-worldv2.pt"

    def __init__(self, initial_classes: Optional[List[str]] = None):
        self.model = None
        self.current_classes: List[str] = []
        self.is_loaded = False
        self._init_model(initial_classes)

    def _init_model(self, initial_classes: Optional[List[str]] = None):
        try:
            from ultralytics import YOLO

            logger.info("Initializing YOLO-World open-vocabulary engine: %s", self.MODEL_NAME)
            self.model = YOLO(self.MODEL_NAME)
            self.is_loaded = True

            classes = initial_classes or [
                c.strip()
                for c in settings.YOLO_WORLD_CLASSES.split(",")
                if c.strip()
            ]
            self.set_classes(classes)
        except Exception as e:
            logger.warning("Could not initialize YOLO-World model: %s", e)
            self.is_loaded = False

    def set_classes(self, classes: List[str]) -> bool:
        """Dynamically update prompt classes without retraining."""
        if not self.is_loaded or self.model is None:
            return False

        clean_classes = [c.strip() for c in classes if c.strip()]
        if not clean_classes:
            clean_classes = ["product", "package", "bottle", "box"]

        try:
            self.model.set_classes(clean_classes)
            self.current_classes = clean_classes
            logger.info("YOLO-World classes updated successfully (%d classes active)", len(self.current_classes))
            return True
        except Exception as e:
            logger.warning("Failed to update YOLO-World classes: %s", e)
            return False

    def get_classes(self) -> List[str]:
        return list(self.current_classes)

    def infer(self, frame: np.ndarray, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Run open-vocabulary inference on frame."""
        if not self.is_loaded or self.model is None:
            return []

        detections: List[Dict[str, Any]] = []
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.inference_mode():
                results = self.model.predict(
                    frame,
                    imgsz=416,
                    conf=confidence_threshold,
                    device=device,
                    verbose=False,
                )

                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for box in boxes:
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        class_name = r.names.get(cls_id, f"Item_{cls_id}")

                        xyxy = box.xyxy[0].cpu().numpy()
                        bx1, by1, bx2, by2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        bw = bx2 - bx1
                        bh = by2 - by1

                        # Filter extreme full frame boxes or tiny noise
                        if (bw * bh) > (frame_area * 0.85) or bw > (frame_w * 0.98) or bh > (frame_h * 0.98):
                            continue
                        if bw < 12 or bh < 12 or (bw * bh) < (frame_area * 0.001):
                            continue

                        detections.append({
                            "tracking_id": None,
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence": round(conf, 2),
                            "x1": bx1,
                            "y1": by1,
                            "x2": bx2,
                            "y2": by2,
                            "engine": "yolo_world",
                        })
        except Exception as e:
            logger.error("YOLO-World inference error: %s", e)

        return detections


class VisionManager:
    """
    Central Coordinator for Vision AI Engines.
    Provides singleton access across FastAPI routes.
    """

    def __init__(self):
        self.gemini_service = GeminiVisionService()
        self.yolo_world_service: Optional[YOLOWorldService] = None
        self.active_engine = settings.VISION_ENGINE  # 'standard', 'yolo_world', 'hybrid'
        self._init_yolo_world_async()

    def _init_yolo_world_async(self):
        try:
            self.yolo_world_service = YOLOWorldService()
        except Exception as e:
            logger.warning("YOLO-World lazy init deferred: %s", e)

    def get_status(self) -> Dict[str, Any]:
        has_gemini = bool(self.gemini_service.get_api_key())
        world_ready = self.yolo_world_service is not None and self.yolo_world_service.is_loaded
        return {
            "active_engine": self.active_engine,
            "gemini": {
                "configured": has_gemini,
                "model": settings.GEMINI_MODEL,
                "cache_size": len(self.gemini_service.cache.cache),
            },
            "yolo_world": {
                "available": world_ready,
                "classes": self.yolo_world_service.get_classes() if world_ready else [],
            },
        }

    def set_engine(self, engine: str) -> None:
        if engine in ["standard", "yolo_world", "hybrid"]:
            self.active_engine = engine
            settings.VISION_ENGINE = engine
            logger.info("Active Vision Engine switched to: %s", engine)


# Global singleton instance
vision_manager = VisionManager()
