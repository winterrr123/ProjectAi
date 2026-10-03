from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.detection import DetectionResult, DetectionSession, ProductCount
from app.models.product import Product


class ProductCRUD:
    @staticmethod
    def list_products(db: Session) -> List[Product]:
        return db.query(Product).order_by(Product.product_name.asc()).all()

    @staticmethod
    def get_or_create_product(db: Session, product_name: str, class_id: int) -> Product:
        product = db.query(Product).filter(Product.product_name == product_name).first()
        if product is None:
            product = Product(product_name=product_name, class_id=class_id)
            db.add(product)
            db.commit()
            db.refresh(product)
        return product


class SessionCRUD:
    @staticmethod
    def create_session(db: Session, session_type: str, video_name: Optional[str] = None) -> DetectionSession:
        session = DetectionSession(session_type=session_type.upper(), video_name=video_name)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def update_session_total(db: Session, session_id: int, total_objects: int) -> None:
        session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
        if session:
            session.total_objects = total_objects
            session.ended_at = func.now()
            db.commit()

    @staticmethod
    def get_sessions(db: Session) -> List[DetectionSession]:
        return db.query(DetectionSession).order_by(DetectionSession.started_at.desc()).all()

    @staticmethod
    def get_session_by_id(db: Session, session_id: int) -> Optional[DetectionSession]:
        return db.query(DetectionSession).filter(DetectionSession.id == session_id).first()

    @staticmethod
    def get_statistics(db: Session) -> Dict[str, object]:
        total_sessions = db.query(DetectionSession).count()
        total_video_sessions = db.query(DetectionSession).filter(DetectionSession.session_type == "VIDEO").count()
        total_camera_sessions = db.query(DetectionSession).filter(DetectionSession.session_type == "CAMERA").count()
        total_objects = db.query(func.coalesce(func.sum(DetectionSession.total_objects), 0)).scalar() or 0
        most_common = (
            db.query(DetectionResult.class_name, func.count(DetectionResult.id).label("count"))
            .group_by(DetectionResult.class_name)
            .order_by(func.count(DetectionResult.id).desc())
            .first()
        )
        avg_confidence = db.query(func.avg(DetectionResult.confidence)).scalar() or 0.0
        classes = (
            db.query(DetectionResult.class_name, func.sum(1).label("count"))
            .group_by(DetectionResult.class_name)
            .all()
        )
        return {
            "total_sessions": total_sessions,
            "total_video_sessions": total_video_sessions,
            "total_camera_sessions": total_camera_sessions,
            "total_objects": int(total_objects),
            "most_common_class": most_common[0] if most_common else None,
            "most_common_count": most_common[1] if most_common else 0,
            "average_confidence": round(float(avg_confidence), 4),
            "class_counts": {name: int(count) for name, count in classes},
        }


class DetectionCRUD:
    @staticmethod
    def save_detection_results(db: Session, session_id: int, detections: List[dict]) -> None:
        for item in detections:
            result = DetectionResult(
                session_id=session_id,
                tracking_id=item.get("tracking_id"),
                class_id=item.get("class_id", 0),
                class_name=item.get("class_name", "Unknown"),
                confidence=float(item.get("confidence", 0.0)),
                x1=float(item.get("x1", 0.0)),
                y1=float(item.get("y1", 0.0)),
                x2=float(item.get("x2", 0.0)),
                y2=float(item.get("y2", 0.0)),
                frame_number=int(item.get("frame_number", 0)),
                timestamp=float(item.get("timestamp", 0.0)),
            )
            db.add(result)
        db.commit()

    @staticmethod
    def save_product_counts(db: Session, session_id: int, counts: Dict[str, int]) -> None:
        for class_name, count in counts.items():
            db.add(ProductCount(session_id=session_id, class_name=class_name, count=count))
        db.commit()

    @staticmethod
    def get_recent_sessions(db: Session, limit: int = 10) -> List[DetectionSession]:
        return db.query(DetectionSession).order_by(DetectionSession.started_at.desc()).limit(limit).all()
