from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.camera_service import CameraService

router = APIRouter(prefix="/api/camera", tags=["camera"])
camera_service = CameraService()


@router.post("/start")
async def start_camera():
    try:
        camera_service.start(0)
        return {"success": True, "message": "Camera started"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot open camera: {exc}")


@router.post("/stop")
async def stop_camera():
    try:
        camera_service.stop()
        return {"success": True, "message": "Camera stopped"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to stop camera: {exc}")
