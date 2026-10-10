import re
import urllib.request
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database.crud import UserCRUD
from app.database.mysql import get_db_session
from app.models.user import User
from app.utils.auth import (
    create_access_token,
    get_current_user,
    verify_google_token,
)
from app.utils.helpers import logger

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class GoogleLoginPayload(BaseModel):
    credential: str


@router.post("/google")
def login_with_google(
    payload: GoogleLoginPayload,
    response: Response,
    db: Session = Depends(get_db_session),
):
    id_info = verify_google_token(payload.credential)
    if not id_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mã định danh Google (ID Token) không hợp lệ hoặc đã hết hạn",
        )

    google_id = id_info.get("sub")
    email = id_info.get("email")
    if not google_id or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không tìm thấy thông tin email từ tài khoản Google",
        )

    full_name = id_info.get("name")
    avatar_url = id_info.get("picture")

    user = UserCRUD.get_or_create_google_user(
        db,
        google_id=google_id,
        email=email,
        full_name=full_name,
        avatar_url=avatar_url,
    )

    token = create_access_token({"sub": str(user.id), "email": user.email})

    max_age_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    response.set_cookie(
        key="access_token",
        value=token,
        max_age=max_age_seconds,
        httponly=True,
        samesite="lax",
    )

    return {
        "success": True,
        "message": "Đăng nhập Google thành công",
        "token": token,
        "user": {
            "id": user.id,
            "google_id": user.google_id,
            "email": user.email,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
        },
    }


class DevLoginPayload(BaseModel):
    name: Optional[str] = "Demo User"
    email: str = "demo@example.com"


@router.post("/dev-login")
def dev_login(
    payload: DevLoginPayload,
    response: Response,
    db: Session = Depends(get_db_session),
):
    user = UserCRUD.get_or_create_google_user(
        db,
        google_id=f"dev_{payload.email}",
        email=payload.email,
        full_name=payload.name,
        avatar_url=None,
    )
    token = create_access_token({"sub": str(user.id), "email": user.email})
    max_age_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    response.set_cookie(
        key="access_token",
        value=token,
        max_age=max_age_seconds,
        httponly=True,
        samesite="lax",
    )
    return {
        "success": True,
        "message": f"Đăng nhập thành công với tài khoản {payload.email}",
        "token": token,
        "user": {
            "id": user.id,
            "google_id": user.google_id,
            "email": user.email,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
        },
    }


@router.get("/me")
def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "success": True,
        "user": {
            "id": current_user.id,
            "google_id": current_user.google_id,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "avatar_url": current_user.avatar_url,
        },
    }


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"success": True, "message": "Đã đăng xuất thành công"}


# Cấu hình dynamic background trích xuất từ Pinterest
_DEFAULT_PINTEREST_PIN = "https://www.pinterest.com/pin/300404237649165148/"
_CACHED_BG_INFO = {
    "pin_url": _DEFAULT_PINTEREST_PIN,
    "background_url": "/static/assets/login_dynamic_bg.gif",
    "online_url": "https://i.pinimg.com/originals/80/cb/9c/80cb9cd9740d77049784c48045764ec4.gif",
    "fallback_url": "https://i.pinimg.com/1200x/80/cb/9c/80cb9cd9740d77049784c48045764ec4.jpg",
    "title": "Ảnh nền động cực đẹp",
    "type": "animated_gif",
}


@router.get("/background")
def get_login_background(pin_url: Optional[str] = None):
    """
    API cung cấp ảnh nền động từ Pinterest link.
    Trả về background_url động để frontend thiết lập hình nền.
    """
    target_pin = pin_url or _DEFAULT_PINTEREST_PIN

    # Nếu gọi pin mặc định, trả về ngay thông tin đã tối ưu và cache
    if target_pin.rstrip("/") == _DEFAULT_PINTEREST_PIN.rstrip("/"):
        return {
            "success": True,
            "pin_url": _DEFAULT_PINTEREST_PIN,
            "background_url": _CACHED_BG_INFO["background_url"],
            "online_url": _CACHED_BG_INFO["online_url"],
            "fallback_url": _CACHED_BG_INFO["fallback_url"],
            "title": _CACHED_BG_INFO["title"],
            "type": _CACHED_BG_INFO["type"],
        }

    # Nếu là pin mới được truyền qua query parameter, fetch và trích xuất URL động
    try:
        req = urllib.request.Request(
            target_pin,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            gif_match = re.search(r'https?://[^\s"\'<>]+\.gif', html)
            img_match = re.search(r'https?://i\.pinimg\.com/(?:originals|1200x|736x)/[^\s"\'<>]+\.(?:jpg|png|webp)', html)

            chosen_url = None
            if gif_match:
                chosen_url = gif_match.group(0)
            elif img_match:
                chosen_url = img_match.group(0)
            else:
                chosen_url = _CACHED_BG_INFO["online_url"]

            return {
                "success": True,
                "pin_url": target_pin,
                "background_url": chosen_url,
                "online_url": chosen_url,
                "fallback_url": _CACHED_BG_INFO["fallback_url"],
                "title": "Pinterest Dynamic Asset",
                "type": "animated_gif" if ".gif" in chosen_url else "image",
            }
    except Exception as exc:
        logger.warning(f"Lỗi khi tải Pinterest pin {target_pin}: {exc}")
        return {
            "success": True,
            "pin_url": target_pin,
            "background_url": _CACHED_BG_INFO["background_url"],
            "online_url": _CACHED_BG_INFO["online_url"],
            "fallback_url": _CACHED_BG_INFO["fallback_url"],
            "title": _CACHED_BG_INFO["title"],
            "type": _CACHED_BG_INFO["type"],
            "note": "Sử dụng cached dynamic asset",
        }

