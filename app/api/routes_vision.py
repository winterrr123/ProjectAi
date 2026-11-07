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
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database.crud import SampleCRUD
from app.database.mysql import get_db_session
from app.models.user import User
from app.services.vision_api_service import vision_manager
from app.utils.auth import get_current_user
from app.utils.helpers import logger

router = APIRouter(prefix="/api/vision", tags=["vision-ai"])


# --- Schemas ---
class IdentifySamplePayload(BaseModel):
    image: str
    api_key: Optional[str] = None

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


@router.post("/identify-sample")
async def identify_sample(
    payload: IdentifySamplePayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Trạm nhận diện mẫu sản phẩm đơn vị (Single-Unit Sample Station):
    1. Nhận diện nhãn hiệu, phân loại trực quan (visual_class) và quy cách qua Gemini Vision.
    2. Tự động lưu mẫu vào cơ sở dữ liệu cho tài khoản người dùng hiện tại.
    """
    raw_b64 = payload.image
    if not raw_b64:
        raise HTTPException(status_code=400, detail="Dữ liệu ảnh mẫu không được để trống.")

    # Keep a copy of data url for sample thumbnail
    sample_thumbnail = raw_b64 if raw_b64.startswith("data:image") else f"data:image/jpeg;base64,{raw_b64}"

    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dữ liệu base64 không hợp lệ: {e}")

    result = vision_manager.gemini_service.identify_sample_product(
        image_bytes=image_bytes,
        api_key=payload.api_key,
    )

    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error", "Lỗi nhận diện mẫu sản phẩm"))

    sample_info = result.get("data", {})
    product_name = sample_info.get("product_name") or "Sản phẩm mẫu"
    brand = sample_info.get("brand") or "Chưa rõ"
    category = sample_info.get("category") or "Hàng hóa"
    visual_class = sample_info.get("visual_class") or "product"
    specs = sample_info.get("specs") or ""

    # Save image to uploads/samples/{uuid}.jpg instead of storing large Base64 in database
    samples_dir = settings.UPLOAD_FOLDER / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_filename = f"{uuid.uuid4().hex}.jpg"
    sample_path = samples_dir / sample_filename
    try:
        with open(sample_path, "wb") as f:
            f.write(image_bytes)
        sample_image_url = f"/api/video/sample-image/{sample_filename}"
    except Exception as exc:
        logger.warning("Could not save sample image file: %s", exc)
        sample_image_url = sample_thumbnail[:500000]

    # Persist in DB
    created_sample = SampleCRUD.create_sample(
        db=db,
        user_id=current_user.id,
        product_name=product_name,
        brand=brand,
        category=category,
        visual_class=visual_class,
        sample_image_url=sample_image_url,
        specs=specs,
    )

    return {
        "success": True,
        "sample": {
            "id": created_sample.id,
            "user_id": created_sample.user_id,
            "product_name": created_sample.product_name,
            "brand": created_sample.brand,
            "category": created_sample.category,
            "visual_class": created_sample.visual_class,
            "specs": created_sample.specs,
            "sample_image_url": created_sample.sample_image_url,
            "created_at": created_sample.created_at.isoformat() if created_sample.created_at else None,
        },
    }


@router.get("/samples")
async def list_user_samples(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Lấy danh sách các mẫu sản phẩm đã lưu trữ của người dùng hiện tại."""
    samples = SampleCRUD.get_user_samples(db, user_id=current_user.id)
    return {
        "success": True,
        "samples": [
            {
                "id": s.id,
                "user_id": s.user_id,
                "product_name": s.product_name,
                "brand": s.brand,
                "category": s.category,
                "visual_class": s.visual_class,
                "specs": s.specs,
                "sample_image_url": s.sample_image_url,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in samples
        ],
    }


@router.delete("/samples/{sample_id}")
async def delete_user_sample(
    sample_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Xóa mẫu sản phẩm khỏi thư viện của người dùng."""
    deleted = SampleCRUD.delete_sample(db, sample_id=sample_id, user_id=current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy mẫu sản phẩm hoặc bạn không có quyền xóa.",
        )
    return {
        "success": True,
        "message": "Đã xóa mẫu sản phẩm thành công.",
    }
