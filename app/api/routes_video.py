from __future__ import annotations

import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.database.crud import DetectionCRUD, SessionCRUD
from app.database.mysql import SessionLocal
from app.services.video_service import VideoProcessingService
from app.utils.helpers import logger, safe_filename

router = APIRouter(prefix="/api/video", tags=["video"])
video_service = VideoProcessingService()


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    if not file.content_type or "video" not in file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid video file.")
    filename = safe_filename(file.filename)
    upload_path = settings.UPLOAD_FOLDER / filename
    try:
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="File is too large")
        with open(upload_path, "wb") as fh:
            fh.write(content)
        logger.info("Video uploaded: %s", filename)
        return {"success": True, "message": "Video uploaded", "filename": filename, "path": str(upload_path)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")


@router.post("/process")
async def process_video(video_name: str):
    db = SessionLocal()
    try:
        if not video_name:
            raise HTTPException(status_code=400, detail="Video name is required")
        file_path = settings.UPLOAD_FOLDER / video_name
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Upload not found")
        session = SessionCRUD.create_session(db, "VIDEO", video_name)
        output_path = str(settings.OUTPUT_FOLDER / f"processed_{session.id}_{os.path.basename(video_name)}")
        result = video_service.process_video(str(file_path), output_path)
        detections = []
        for item in result.get("detections", []):
            detections.append({
                "tracking_id": item.get("tracking_id"),
                "class_id": item.get("class_id", 0),
                "class_name": item.get("class_name", "Unknown"),
                "confidence": float(item.get("confidence", 0.0)),
                "x1": float(item.get("x1", 0.0)),
                "y1": float(item.get("y1", 0.0)),
                "x2": float(item.get("x2", 0.0)),
                "y2": float(item.get("y2", 0.0)),
                "frame_number": int(item.get("frame_number", 0)),
                "timestamp": float(item.get("timestamp", 0.0)),
            })
        DetectionCRUD.save_detection_results(db, session.id, detections)
        DetectionCRUD.save_product_counts(db, session.id, result.get("by_class", {}))
        SessionCRUD.update_session_total(db, session.id, result.get("total_objects", 0))
        return {"success": True, "message": "Video processed successfully", "session_id": session.id, "output_video": output_path, "stats": result}
    except SQLAlchemyError as exc:
        logger.exception("Database issue during processing: %s", exc)
        raise HTTPException(status_code=500, detail="Database error while saving results")
    except Exception as exc:
        logger.exception("Error while processing video: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@router.get("/result/{session_id}")
async def get_video_result(session_id: int):
    db = SessionLocal()
    try:
        session = SessionCRUD.get_session_by_id(db, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"success": True, "session": {"id": session.id, "video_name": session.video_name, "total_objects": session.total_objects, "session_type": session.session_type}}
    finally:
        db.close()
