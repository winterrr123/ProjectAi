from __future__ import annotations

from fastapi import APIRouter

from app.database.crud import SessionCRUD
from app.database.mysql import SessionLocal

router = APIRouter(prefix="/api", tags=["statistics"])


@router.get("/sessions")
async def list_sessions():
    db = SessionLocal()
    try:
        sessions = SessionCRUD.get_sessions(db)
        return {"success": True, "sessions": [{"id": s.id, "session_type": s.session_type, "video_name": s.video_name, "started_at": s.started_at.isoformat() if s.started_at else None, "ended_at": s.ended_at.isoformat() if s.ended_at else None, "total_objects": s.total_objects} for s in sessions]}
    finally:
        db.close()


@router.get("/sessions/{session_id}")
async def get_session(session_id: int):
    db = SessionLocal()
    try:
        session = SessionCRUD.get_session_by_id(db, session_id)
        if session is None:
            return {"success": False, "message": "Session not found"}
        return {"success": True, "session": {"id": session.id, "session_type": session.session_type, "video_name": session.video_name, "started_at": session.started_at.isoformat() if session.started_at else None, "ended_at": session.ended_at.isoformat() if session.ended_at else None, "total_objects": session.total_objects}}
    finally:
        db.close()


@router.get("/statistics")
async def statistics():
    db = SessionLocal()
    try:
        data = SessionCRUD.get_statistics(db)
        return {"success": True, "statistics": data}
    finally:
        db.close()


@router.get("/products")
async def products():
    from app.database.crud import ProductCRUD

    db = SessionLocal()
    try:
        data = ProductCRUD.list_products(db)
        return {"success": True, "products": [{"id": p.id, "product_name": p.product_name, "class_id": p.class_id, "created_at": p.created_at.isoformat()} for p in data]}
    finally:
        db.close()
