from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import jwt
from sqlalchemy.orm import Session

from app.config import settings
from app.database.crud import UserCRUD
from app.database.mysql import get_db_session
from app.models.user import User
from app.utils.helpers import logger


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError as exc:
        logger.debug("JWT decode failed: %s", exc)
        return None


def verify_google_token(credential: str) -> Optional[dict]:
    try:
        request_adapter = google_requests.Request()
        audience = settings.GOOGLE_CLIENT_ID if settings.GOOGLE_CLIENT_ID else None
        # Cho phép độ lệch đồng hồ 120 giây (clock skew) giữa máy tính cục bộ và máy chủ Google
        id_info = id_token.verify_oauth2_token(
            credential,
            request_adapter,
            audience=audience,
            clock_skew_in_seconds=120,
        )
        return id_info
    except Exception as exc:
        logger.warning("Google ID token verification failed: %s", exc)
        return None


def get_token_from_request(request: Request) -> Optional[str]:
    # Check Cookie first
    token = request.cookies.get("access_token")
    if token:
        return token
    # Check Authorization header: Bearer <token>
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]
    return None


def get_current_user_optional(
    request: Request, db: Session = Depends(get_db_session)
) -> Optional[User]:
    token = get_token_from_request(request)
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    try:
        user_id_int = int(user_id)
    except ValueError:
        return None
    user = UserCRUD.get_by_id(db, user_id_int)
    return user


def get_current_user(
    request: Request, db: Session = Depends(get_db_session)
) -> User:
    user = get_current_user_optional(request, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yêu cầu đăng nhập để thực hiện thao tác này",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_user_page(
    request: Request, db: Session = Depends(get_db_session)
) -> User:
    user = get_current_user_optional(request, db)
    if not user:
        # Redirect to login
        next_url = request.url.path
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": f"/login?next={next_url}"},
        )
    return user
