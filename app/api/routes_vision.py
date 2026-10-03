"""
==============================================================================
PRODUCT VISION - VISION AI ROUTER (GEMINI FLASH & YOLO-WORLD)
FastAPI Endpoints for Open-Vocabulary Detection & Multimodal Brand Intelligence
==============================================================================
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services.vision_api_service import vision_manager
from app.utils.helpers import logger

router = APIRouter(prefix="/api/vision", tags=["vision-ai"])


# --- Schemas ---
class EngineConfigPayload(BaseModel):
    engine: str  # 'standard', 'yolo_world', 'hybrid'
    classes: Optional[List[str]] = None


class GeminiKeyPayload(BaseModel):
    api_key: str
    persist_to_env: Optional[bool] = True


class IdentifyProductPayload(BaseModel):
    image: str  # Base64 image (data:image/jpeg;base64,... or pure base64)
    prompt: Optional[str] = None
    api_key: Optional[str] = None


class SceneAnalysisPayload(BaseModel):
    image: str
    api_key: Optional[str] = None


class CustomClassesPayload(BaseModel):
    classes: List[str]


# --- Endpoints ---
@router.get("/status")
async def get_vision_status():
    """Lấy trạng thái cấu hình của hệ thống Vision AI (Gemini & YOLO-World)."""
    return {
        "success": True,
        "status": vision_manager.get_status(),
    }


@router.post("/set-engine")
async def set_vision_engine(payload: EngineConfigPayload):
    """Chuyển đổi công cụ nhận diện: standard (mô hình đã train), yolo_world (vạn vật), hybrid."""
    if payload.engine not in ["standard", "yolo_world", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail="Engine không hợp lệ. Chọn 'standard', 'yolo_world' hoặc 'hybrid'.",
        )

    vision_manager.set_engine(payload.engine)

    # If classes provided, update YOLO-World
    if payload.classes and vision_manager.yolo_world_service:
        vision_manager.yolo_world_service.set_classes(payload.classes)

    return {
        "success": True,
        "message": f"Đã chuyển sang chế độ nhận diện: {payload.engine}",
        "status": vision_manager.get_status(),
    }


@router.post("/set-classes")
async def set_custom_classes(payload: CustomClassesPayload):
    """Cập nhật danh sách từ khóa nhận diện vạn vật theo thời gian thực (Zero-Shot)."""
    if not vision_manager.yolo_world_service:
        raise HTTPException(status_code=503, detail="Mô hình YOLO-World chưa sẵn sàng.")

    success = vision_manager.yolo_world_service.set_classes(payload.classes)
    if not success:
        raise HTTPException(status_code=500, detail="Không thể cập nhật danh sách nhãn.")

    return {
        "success": True,
        "message": f"Đã cập nhật {len(payload.classes)} nhãn nhận diện mới!",
        "classes": vision_manager.yolo_world_service.get_classes(),
    }


@router.post("/gemini/test-key")
async def test_gemini_key(payload: GeminiKeyPayload):
    """Kiểm tra tính hợp lệ của Google Gemini API Key."""
    result = vision_manager.gemini_service.test_key(payload.api_key)
    return {
        "success": result.get("valid", False),
        "message": result.get("message", ""),
    }


@router.post("/gemini/save-key")
async def save_gemini_key(payload: GeminiKeyPayload):
    """Lưu Gemini API Key vào hệ thống và file .env."""
    key = payload.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API Key không được để trống.")

    # Test key first
    test_res = vision_manager.gemini_service.test_key(key)
    if not test_res.get("valid"):
        raise HTTPException(
            status_code=400,
            detail=f"API Key không hợp lệ: {test_res.get('message')}",
        )

    vision_manager.gemini_service.set_api_key(key)
    settings.GEMINI_API_KEY = key

    # Persist to .env
    if payload.persist_to_env:
        try:
            env_file = settings.PROJECT_ROOT / ".env"
            content = ""
            if env_file.exists():
                content = env_file.read_text(encoding="utf-8")

            if "GEMINI_API_KEY=" in content:
                content = re.sub(r"GEMINI_API_KEY=.*", f"GEMINI_API_KEY={key}", content)
            else:
                content += f"\nGEMINI_API_KEY={key}\n"

            env_file.write_text(content, encoding="utf-8")
            logger.info("GEMINI_API_KEY saved to .env")
        except Exception as e:
            logger.warning("Could not persist GEMINI_API_KEY to .env: %s", e)

    return {
        "success": True,
        "message": "Đã lưu Gemini API Key thành công!",
        "status": vision_manager.get_status(),
    }


@router.post("/identify")
async def identify_product(payload: IdentifyProductPayload):
    """
    Nhận diện chi tiết sản phẩm qua Gemini Vision API:
    Tên tiếng Việt, thương hiệu (Brand), loại hàng, bao bì, quy cách và barcode.
    """
    raw_b64 = payload.image
    if not raw_b64:
        raise HTTPException(status_code=400, detail="Dữ liệu ảnh không được để trống.")

    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dữ liệu base64 không hợp lệ: {e}")

    result = vision_manager.gemini_service.identify_product(
        image_bytes=image_bytes,
        custom_prompt=payload.prompt,
        api_key=payload.api_key,
    )

    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error", "Lỗi nhận diện Gemini"))

    return result


@router.post("/analyze-scene")
async def analyze_scene(payload: SceneAnalysisPayload):
    """Phân tích toàn cảnh hình ảnh / khung hình video qua Gemini Vision."""
    raw_b64 = payload.image
    if not raw_b64:
        raise HTTPException(status_code=400, detail="Dữ liệu ảnh không được để trống.")

    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dữ liệu base64 không hợp lệ: {e}")

    result = vision_manager.gemini_service.analyze_full_scene(
        image_bytes=image_bytes,
        api_key=payload.api_key,
    )

    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error", "Lỗi phân tích Gemini"))

    return result
