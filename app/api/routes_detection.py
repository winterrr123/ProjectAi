from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.camera_service import CameraService

router = APIRouter(prefix="/api", tags=["detection"])
camera_service = CameraService()


@router.get("/health")
async def health_check():
    return {"success": True, "message": "Server is healthy"}
