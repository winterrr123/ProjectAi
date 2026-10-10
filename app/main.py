import asyncio
import base64
import json
import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

import cv2
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.cors import CORSMiddleware

from app.api.routes_auth import router as auth_router
from app.api.routes_vision import router as vision_router
from app.config import settings
from app.database.crud import DetectionCRUD, ProductCRUD, SessionCRUD
from app.database.mysql import SessionLocal, init_db
from app.models.user import User
from app.utils.auth import decode_access_token, get_current_user, get_token_from_request
from app.services.camera_service import CameraService
from app.services.tracking_service import ByteTrackService
from app.services.video_service import VideoProcessingService
from app.schemas.detection_schema import (
    CameraSessionPayload,
    DetectionRequest,
    DetectionResponse,
    FrameDetectionPayload,
    SaveLiveSessionPayload,
)
from app.utils.helpers import ensure_log_dir, is_valid_video, logger, safe_filename

ensure_log_dir()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
        logger.info("Database initialized")
        if not os.path.exists(settings.MODEL_PATH):
            logger.warning("Model file not found at %s", settings.MODEL_PATH)
        yield
    except Exception as exc:
        logger.exception("Startup error: %s", exc)
        raise


app = FastAPI(title="AI Product Detection System", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(vision_router)

app.mount("/static", StaticFiles(directory=str(settings.FRONTEND_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(settings.FRONTEND_DIR)), name="assets")


templates = Jinja2Templates(directory=str(settings.FRONTEND_DIR))
video_service = VideoProcessingService()
camera_service = CameraService(yolo_service=video_service.yolo_service)


def is_authenticated(request: Request) -> bool:
    token = get_token_from_request(request)
    if not token:
        return False
    payload = decode_access_token(token)
    return payload is not None and bool(payload.get("sub"))


@app.get("/login")
@app.get("/login.html")
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "google_client_id": settings.GOOGLE_CLIENT_ID})


@app.get("/")
@app.get("/index.html")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload")
@app.get("/upload.html")
async def upload_page(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/camera")
@app.get("/camera.html")
async def camera_page(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)
    return templates.TemplateResponse("camera.html", {"request": request})


@app.get("/history")
@app.get("/history.html")
async def history_page(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/statistics")
@app.get("/statistics.html")
async def statistics_page(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)
    return templates.TemplateResponse("statistics.html", {"request": request})


@app.get("/api/health")
async def health_check():
    return {"success": True, "message": "Server is healthy", "model_path": settings.MODEL_PATH}


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    if not file.content_type or "video" not in file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    unique_name = safe_filename(file.filename)
    upload_path = settings.UPLOAD_FOLDER / unique_name

    try:
        with open(upload_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        size = upload_path.stat().st_size
        if size > settings.MAX_UPLOAD_SIZE:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="File too large")
        logger.info("Video uploaded successfully: %s (%s bytes)", upload_path, size)
        return {
            "success": True,
            "message": "Video uploaded",
            "filename": unique_name,
            "path": str(upload_path),
            "video_url": f"/api/video/file/{unique_name}",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")


@app.get("/api/video/file/{filename}")
async def serve_video_file(filename: str):
    file_path = settings.UPLOAD_FOLDER / filename
    if not file_path.exists():
        # Fallback to project root if sample file is requested
        root_path = settings.PROJECT_ROOT / filename
        if root_path.exists():
            file_path = root_path
        else:
            raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path=str(file_path), media_type="video/mp4")


@app.post("/api/detect/frame")
async def detect_frame(payload: FrameDetectionPayload):
    try:
        is_camera = (payload.source == "camera")
        target_service = camera_service if is_camera else video_service

        if payload.reset:
            if hasattr(target_service, 'reset'):
                target_service.reset()
            else:
                target_service.counter.reset()
                if hasattr(target_service, 'yolo_service'):
                    target_service.yolo_service.reset_tracker()
                elif hasattr(target_service, 'model'):
                    target_service.model.reset_tracker()
                if hasattr(target_service, '_smooth_boxes'):
                    target_service._smooth_boxes.clear()
                if hasattr(target_service, 'tracker'):
                    target_service.tracker = ByteTrackService()

        image_data = payload.image
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image data")

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            decoded = base64.b64decode(image_data)
            np_arr = np.frombuffer(decoded, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as decode_err:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {decode_err}")

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image frame data")

        result = target_service.process_frame(frame, frame_number=payload.frame_number)
        return {
            "success": True,
            "detections": result.get("detections", []),
            "counts": result.get("counts", {}),
            "total": result.get("total", 0),
            "frame_number": payload.frame_number,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Frame detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/camera/detect-frame")
async def camera_detect_frame(payload: FrameDetectionPayload):
    payload.source = "camera"
    return await detect_frame(payload)


@app.post("/api/detect/reset")
async def detect_reset():
    # Reset video service (uses YOLO built-in tracker)
    video_service.reset()
    # Reset camera service
    camera_service.reset()
    return {"success": True, "message": "Tracker and counter reset"}


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    High-performance WebSocket endpoint for real-time frame detection.
    Eliminates HTTP round-trip latency and TCP handshake overhead.
    """
    await websocket.accept()
    source = "video"
    frame_number = 0
    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    continue

                msg_type = data.get("type", "frame")
                source = data.get("source", source)
                target_service = camera_service if source == "camera" else video_service

                if msg_type == "reset" or data.get("reset"):
                    target_service.reset()
                    await websocket.send_json({"success": True, "type": "reset_ack"})
                    continue

                image_data = data.get("image", "")
                if not image_data:
                    continue

                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]

                try:
                    np_arr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception:
                    continue

                frame_number = data.get("frame_number", frame_number + 1)
            elif "bytes" in message:
                raw_bytes = message["bytes"]
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                frame_number += 1
                target_service = camera_service if source == "camera" else video_service
            else:
                continue

            if frame is None:
                continue

            result = target_service.process_frame(frame, frame_number=frame_number)
            await websocket.send_json({
                "success": True,
                "detections": result.get("detections", []),
                "counts": result.get("counts", {}),
                "total": result.get("total", 0),
                "frame_number": frame_number,
            })
    except WebSocketDisconnect:
        logger.info("WebSocket detection client disconnected")
    except Exception as exc:
        logger.warning("WebSocket handler note: %s", exc)


@app.post("/api/video/save-session")
async def save_live_session(payload: SaveLiveSessionPayload, current_user: User = Depends(get_current_user)):
    db = None
    try:
        db = SessionLocal()
        session = SessionCRUD.create_session(
            db, payload.session_type, payload.video_name, user_id=current_user.id
        )
        SessionCRUD.update_session_total(db, session.id, payload.total_objects)
        if payload.by_class:
            DetectionCRUD.save_product_counts(db, session.id, payload.by_class)
        return {
            "success": True,
            "message": "Live session saved successfully",
            "session_id": session.id,
        }
    except SQLAlchemyError as exc:
        logger.exception("Database error saving live session: %s", exc)
        raise HTTPException(status_code=500, detail="Database error while saving session")
    except Exception as exc:
        logger.exception("Failed to save live session: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if db is not None:
            db.close()


@app.post("/api/video/process")
async def process_video(payload: DetectionRequest, current_user: User = Depends(get_current_user)):
    db = None
    try:
        db = SessionLocal()
        session = SessionCRUD.create_session(
            db, payload.session_type, payload.video_name, user_id=current_user.id
        )
        video_name = payload.video_name or "uploaded_video.mp4"
        if not os.path.exists(settings.UPLOAD_FOLDER / video_name):
            raise HTTPException(status_code=404, detail="Upload not found")

        output_path = str(settings.OUTPUT_FOLDER / f"processed_{session.id}_{os.path.basename(video_name)}")
        result = await asyncio.to_thread(video_service.process_video, str(settings.UPLOAD_FOLDER / video_name), output_path)
        output_filename = os.path.basename(result.get("output_video") or output_path)

        DetectionCRUD.save_detection_results(
            db,
            session.id,
            [
                {
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
                }
                for item in result.get("detections", [])
            ],
        )
        DetectionCRUD.save_product_counts(db, session.id, result.get("by_class", {}))
        SessionCRUD.update_session_total(db, session.id, result.get("total_objects", 0))

        return {
            "success": True,
            "message": "Video processed successfully",
            "session_id": session.id,
            "output_video": output_filename,
            "output_path": output_path,
            "stats": result,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except SQLAlchemyError as exc:
        logger.exception("Database error during video processing: %s", exc)
        raise HTTPException(status_code=500, detail="Database error while saving results")
    except Exception as exc:
        logger.exception("Video processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if db is not None:
            db.close()


@app.get("/api/video/process/progress")
async def get_video_process_progress():
    return getattr(video_service, "progress", {"percent": 0, "status": "idle", "message": ""})


@app.post("/api/camera/start")
async def camera_start(payload: CameraSessionPayload):
    if payload.action != "start":
        raise HTTPException(status_code=400, detail="Only start action is supported")
    try:
        # Reset detection and tracking state for live camera session
        video_service.counter.reset()
        video_service.tracker = ByteTrackService()
        logger.info("Camera start request accepted (browser client stream)")
        return {"success": True, "message": "Camera session initialized"}
    except Exception as exc:
        logger.exception("Cannot initialize camera session: %s", exc)
        return {"success": True, "message": "Camera session initialized"}


@app.post("/api/camera/stop")
async def camera_stop(payload: CameraSessionPayload):
    if payload.action != "stop":
        raise HTTPException(status_code=400, detail="Only stop action is supported")
    try:
        logger.info("Camera stop request accepted")
        return {"success": True, "message": "Camera stopped"}
    except Exception as exc:
        logger.exception("Camera stop error: %s", exc)
        return {"success": True, "message": "Camera stopped"}


@app.get("/api/sessions")
async def list_sessions(current_user: User = Depends(get_current_user)):
    db = None
    try:
        db = SessionLocal()
        sessions = SessionCRUD.get_sessions(db, user_id=current_user.id)
        return {
            "success": True,
            "sessions": [
                {
                    "id": s.id,
                    "session_type": s.session_type,
                    "video_name": s.video_name,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                    "total_objects": s.total_objects,
                }
                for s in sessions
            ],
        }
    except SQLAlchemyError as exc:
        logger.exception("Database error while listing sessions: %s", exc)
        return {"success": False, "message": "Database unavailable", "sessions": []}
    finally:
        if db is not None:
            db.close()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: int, current_user: User = Depends(get_current_user)):
    db = None
    try:
        db = SessionLocal()
        session = SessionCRUD.get_session_by_id(db, session_id, user_id=current_user.id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "success": True,
            "session": {
                "id": session.id,
                "session_type": session.session_type,
                "video_name": session.video_name,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "total_objects": session.total_objects,
            },
        }
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.exception("Database error while fetching session: %s", exc)
        return {"success": False, "message": "Database unavailable"}
    finally:
        if db is not None:
            db.close()


@app.get("/api/statistics")
async def statistics(current_user: User = Depends(get_current_user)):
    db = None
    try:
        db = SessionLocal()
        stats = SessionCRUD.get_statistics(db, user_id=current_user.id)
        return {"success": True, "statistics": stats}
    except SQLAlchemyError as exc:
        logger.exception("Database error while fetching statistics: %s", exc)
        return {"success": False, "message": "Database unavailable", "statistics": {}}
    finally:
        if db is not None:
            db.close()


@app.get("/api/products")
async def get_products():
    db = None
    try:
        db = SessionLocal()
        products = ProductCRUD.list_products(db)
        return {
            "success": True,
            "products": [
                {"id": p.id, "product_name": p.product_name, "class_id": p.class_id, "created_at": p.created_at.isoformat()}
                for p in products
            ],
        }
    except SQLAlchemyError as exc:
        logger.exception("Database error while fetching products: %s", exc)
        return {"success": False, "message": "Database unavailable", "products": []}
    finally:
        if db is not None:
            db.close()


@app.get("/api/video/result/{session_id}")
async def get_video_result(session_id: int, current_user: User = Depends(get_current_user)):
    db = None
    try:
        db = SessionLocal()
        session = SessionCRUD.get_session_by_id(db, session_id, user_id=current_user.id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "success": True,
            "session": {
                "id": session.id,
                "video_name": session.video_name,
                "total_objects": session.total_objects,
                "session_type": session.session_type,
            },
        }
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.exception("Database error while fetching result: %s", exc)
        return {"success": False, "message": "Database unavailable"}
    finally:
        if db is not None:
            db.close()


@app.get("/api/video/output/{filename}")
async def serve_output(filename: str):
    file_path = settings.OUTPUT_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(path=str(file_path), media_type="video/mp4")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "message": exc.detail})


app.mount("/frontend", StaticFiles(directory=str(settings.FRONTEND_DIR)), name="frontend")
