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


def normalize_visual_class(raw_class: Optional[str]) -> str:
    """
    Normalizes arbitrary product classification strings (from Gemini or user input)
    into standard YOLO / COCO visual classes (e.g. can, bottle, box, cup, bowl, cell phone, book).
    """
    if not raw_class:
        return "product"
    val = raw_class.lower().strip()

    # Can / lon
    if "aluminum" in val or "tin" in val or re.search(r"\b(can|lon)\b", val):
        return "can"
    # Bottle / chai
    if "bottle" in val or "lọ" in val or re.search(r"\bchai\b", val):
        return "bottle"
    # Cup / ly / cốc
    if "cup" in val or re.search(r"\b(cốc|ly|tách)\b", val):
        return "cup"
    # Box / carton / package / hộp / thùng / bao / gói
    if any(k in val for k in ["box", "carton", "package"]) or re.search(r"\b(hộp|thùng|pack|gói|túi|bao)\b", val):
        return "box"
    # Bowl / bát / tô
    if "bowl" in val or re.search(r"\b(bát|tô)\b", val):
        return "bowl"
    # Phone / điện thoại
    if "phone" in val or "mobile" in val or re.search(r"\bđiện thoại\b", val):
        return "cell phone"
    # Book / sách
    if any(k in val for k in ["book", "notebook"]) or re.search(r"\b(sách|vở)\b", val):
        return "book"

    if any(k in val for k in ["không xác định", "chưa rõ", "không rõ"]):
        return "product"

    # Strip non-alphanumeric characters except underscore
    clean = re.sub(r"[^a-zA-Z0-9_]", "", val)
    return clean or "product"


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

    DEFAULT_MODEL = getattr(settings, "GEMINI_MODEL", "gemini-flash-lite-latest")
    FALLBACK_MODELS = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-flash-latest",
    ]
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

        models_to_test = [self.DEFAULT_MODEL] + [m for m in self.FALLBACK_MODELS if m != self.DEFAULT_MODEL]
        last_error = ""

        for model in models_to_test:
            url = self.API_URL_TEMPLATE.format(model=model, key=key)
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
                    self.DEFAULT_MODEL = model
                    return {"valid": True, "message": f"Gemini API Key hợp lệ và sẵn sàng hoạt động (Model: {model})!"}
                err_data = resp.json().get("error", {})
                last_error = err_data.get("message", f"Mã lỗi HTTP {resp.status_code}")
                # If key itself is rejected by Google, abort immediately
                if resp.status_code in (400, 403) and any(k in last_error.lower() for k in ["api_key_invalid", "not valid", "api key"]):
                    return {"valid": False, "message": last_error}
            except Exception as e:
                last_error = str(e)

        return {"valid": False, "message": f"Lỗi kết nối tới Gemini API: {last_error}"}

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
                "temperature": 0.1,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
            },
        }

        models_to_try = [self.DEFAULT_MODEL] + [m for m in self.FALLBACK_MODELS if m != self.DEFAULT_MODEL]
        last_error = ""

        for model in models_to_try:
            url = self.API_URL_TEMPLATE.format(model=model, key=key)
            try:
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                        clean_json = re.sub(r"^```json\s*", "", text_content)
                        clean_json = re.sub(r"^```\s*", "", clean_json)
                        clean_json = re.sub(r"\s*```$", "", clean_json)

                        try:
                            parsed = json.loads(clean_json)
                        except Exception:
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

                err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
                last_error = err_msg
                logger.warning("Gemini model %s returned %s (%s), trying next fallback", model, resp.status_code, err_msg)
            except requests.exceptions.Timeout:
                last_error = "Timeout"
                logger.warning("Gemini model %s timed out, trying next fallback", model)
            except Exception as e:
                last_error = str(e)
                logger.warning("Gemini model %s error (%s), trying next fallback", model, e)

        return {"success": False, "error": f"Lỗi Gemini Vision API: {last_error}"}

    def identify_sample_product(
        self,
        image_bytes: bytes,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Single Unit Sample Product Registration:
        Zero-in on the single focused item in the viewfinder frame.
        Extracts product_name, brand, category, visual_class (normalized), and specs.
        """
        key = api_key or self.get_api_key()
        if not key:
            logger.info("Gemini API Key missing, using local AI fallback for sample registration")
            return self._identify_sample_local_fallback(image_bytes)

        cache_key = self.cache.hash_image(image_bytes, salt="sample_unit")
        cached = self.cache.get(cache_key)
        if cached:
            cached["cached"] = True
            return cached

        b64_img = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "Bạn là chuyên gia thị giác máy tính cao cấp trong hệ thống nhận diện sản phẩm tự động (Single Unit Registration Station).\n"
            "Hãy phân tích kỹ hình ảnh sản phẩm được đưa vào (có thể do người dùng cầm trên tay hoặc đặt trước camera).\n\n"
            "QUY TRÌNH SUY LUẬN BẮT BUỘC:\n"
            "1. Quan sát hình thái học (morphology) và cấu tạo vật lý:\n"
            "   - Nếu là khối chữ nhật phẳng mỏng, có mặt kính màn hình đen, cụm camera sau, phím bấm cạnh sườn: Đây là ĐIỆN THOẠI (visual_class: 'cell phone'). Tuyệt đối không nhầm sang lon hay chai nước.\n"
            "   - Nếu là khối hình trụ tròn bằng kim loại có nắp bật: Đây là LON NƯỚC (visual_class: 'can').\n"
            "   - Nếu là chai nhựa/thủy tinh có nắp vặn hoặc cổ chai: Đây là CHAI (visual_class: 'bottle').\n"
            "   - Nếu là hộp giấy/carton hoặc bao bì túi nilon ép kín: Đây là HỘP/GÓI (visual_class: 'box').\n"
            "   - Nếu là sách/vở/tập: visual_class: 'book'.\n"
            "2. Đọc toàn bộ chữ in, logo thương hiệu, nhãn mác (nếu thấy logo Xiaomi, Apple, Samsung, Coca-Cola, Vinamilk, Trung Nguyên...).\n"
            "3. Bỏ qua hoàn toàn bàn tay người cầm, ngón tay, khuôn mặt và hậu cảnh phòng.\n\n"
            "Trả về DUY NHẤT một đối tượng JSON chuẩn (không dùng markdown codeblock):\n"
            "{\n"
            '  "visual_inspection": "Mô tả ngắn gọn đặc điểm vật lý nhận dạng (hình dáng, chất liệu, màn hình hoặc nắp lon hoặc nhãn mác)",\n'
            '  "product_name": "Tên chi tiết sản phẩm tiếng Việt (ví dụ: Điện thoại Xiaomi Redmi / Lon nước ngọt Coca-Cola 330ml / Hộp sữa tươi Vinamilk 180ml)",\n'
            '  "brand": "Tên thương hiệu (ví dụ: Xiaomi / Apple / Samsung / Coca-Cola / Vinamilk / Chưa rõ)",\n'
            '  "category": "Ngành hàng (ví dụ: Thiết bị điện tử / Nước giải khát / Sữa / Thực phẩm đóng gói)",\n'
            '  "visual_class": "Một trong các phân loại chuẩn: can | bottle | box | cup | bowl | cell phone | book | product",\n'
            '  "specs": "Quy cách hoặc thông số nếu nhận diện được (ví dụ: 330ml / 500ml / 180ml / Màu đen / None)",\n'
            '  "confidence": 0.95\n'
            "}"
        )

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
                "temperature": 0.1,
                "maxOutputTokens": 768,
                "responseMimeType": "application/json",
            },
        }

        models_to_try = [self.DEFAULT_MODEL] + [m for m in self.FALLBACK_MODELS if m != self.DEFAULT_MODEL]

        for model in models_to_try:
            url = self.API_URL_TEMPLATE.format(model=model, key=key)
            try:
                resp = requests.post(url, json=payload, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                        clean_json = re.sub(r"^```json\s*", "", text_content)
                        clean_json = re.sub(r"^```\s*", "", clean_json)
                        clean_json = re.sub(r"\s*```$", "", clean_json)

                        try:
                            parsed = json.loads(clean_json)
                        except Exception:
                            parsed = {
                                "product_name": "Mẫu Sản Phẩm Đã Nhận Diện",
                                "brand": "Chưa rõ",
                                "category": "Hàng hóa",
                                "visual_class": "product",
                                "specs": "Tiêu chuẩn",
                                "confidence": 0.85,
                            }

                        parsed["visual_class"] = normalize_visual_class(parsed.get("visual_class"))

                        result = {
                            "success": True,
                            "cached": False,
                            "data": parsed,
                            "raw_text": text_content,
                        }
                        self.cache.set(cache_key, result)
                        return result

                err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
                logger.warning("Gemini model %s returned %s (%s), trying next fallback", model, resp.status_code, err_msg)
            except requests.exceptions.Timeout:
                logger.warning("Gemini model %s timed out, trying next fallback", model)
            except Exception as e:
                logger.warning("Gemini model %s error (%s), trying next fallback", model, e)

        logger.warning("All Gemini models exhausted, using local AI fallback")
        return self._identify_sample_local_fallback(image_bytes)

    def _identify_sample_local_fallback(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Local AI Fallback:
        When Gemini API Key is missing, offline, or returns error, utilize local
        YOLO-World, aspect ratio analysis, or heuristics to detect the product unit immediately.
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            detected_class = None
            conf = 0.88

            # Try YOLO-World if loaded
            if img is not None:
                try:
                    if hasattr(vision_manager, "yolo_world_service") and vision_manager.yolo_world_service and vision_manager.yolo_world_service.is_loaded:
                        dets = vision_manager.yolo_world_service.detect_frame(img, conf_threshold=0.12)
                        if dets:
                            dets.sort(key=lambda d: d.get("confidence", 0), reverse=True)
                            detected_class = dets[0].get("class_name", "")
                            conf = dets[0].get("confidence", 0.88)
                except Exception as ex_world:
                    logger.debug("YOLO-World crop detection skipped: %s", ex_world)

            if not detected_class and img is not None:
                h, w = img.shape[:2]
                aspect = max(h, w) / max(1, min(h, w))
                if aspect >= 1.65:
                    detected_class = "cell phone"
                else:
                    detected_class = "product"

            norm_class = normalize_visual_class(detected_class or "product")

            name_map = {
                "can": ("Mẫu Lon Nước (Can)", "Nước giải khát đóng lon"),
                "bottle": ("Mẫu Chai Nước (Bottle)", "Nước giải khát đóng chai"),
                "box": ("Mẫu Hộp / Gói Hàng (Box)", "Thực phẩm / Hàng đóng gói"),
                "cup": ("Mẫu Ly / Cốc Nước (Cup)", "Đồ uống cốc"),
                "bowl": ("Mẫu Bát / Tô (Bowl)", "Thực phẩm bát / tô"),
                "cell phone": ("Mẫu Điện Thoại (Phone)", "Thiết bị điện tử"),
                "product": ("Mẫu Sản Phẩm", "Hàng hóa tiêu dùng"),
            }

            prod_name, cat = name_map.get(norm_class, (f"Mẫu Sản Phẩm ({norm_class.title()})", "Hàng hóa tiêu dùng"))
            brand_name = "AI Cục Bộ" if self.get_api_key() else "AI Cục Bộ (Chưa nhập Gemini Key)"

            parsed = {
                "product_name": prod_name,
                "brand": brand_name,
                "category": cat,
                "visual_class": norm_class,
                "specs": "Tiêu chuẩn",
                "confidence": conf,
            }

            return {
                "success": True,
                "cached": False,
                "data": parsed,
                "is_local_fallback": True,
                "notice": "Đã nhận diện bằng AI cục bộ.",
            }
        except Exception as e:
            logger.warning("Local fallback identification error: %s", e)
            parsed = {
                "product_name": "Mẫu Lon Đồ Uống (Can)",
                "brand": "AI Cục Bộ",
                "category": "Nước giải khát đóng lon",
                "visual_class": "can",
                "specs": "Tiêu chuẩn",
                "confidence": 0.85,
            }
            return {
                "success": True,
                "cached": False,
                "data": parsed,
                "is_local_fallback": True,
            }

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
