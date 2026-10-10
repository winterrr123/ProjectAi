from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.detection import DetectionResult, DetectionSession, ProductCount
from app.models.product import Product
from app.models.user import User


class UserCRUD:
    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_google_id(db: Session, google_id: str) -> Optional[User]:
        return db.query(User).filter(User.google_id == google_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def get_or_create_google_user(
        db: Session,
        google_id: str,
        email: str,
        full_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> User:
        user = db.query(User).filter((User.google_id == google_id) | (User.email == email)).first()
        if user:
            updated = False
            if full_name and user.full_name != full_name:
                user.full_name = full_name
                updated = True
            if avatar_url and user.avatar_url != avatar_url:
                user.avatar_url = avatar_url
                updated = True
            if user.google_id != google_id:
                user.google_id = google_id
                updated = True
            if updated:
                db.commit()
                db.refresh(user)
            return user

        user = User(
            google_id=google_id,
            email=email,
            full_name=full_name,
            avatar_url=avatar_url,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


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
    def create_session(
        db: Session, session_type: str, video_name: Optional[str] = None, user_id: int = 1
    ) -> DetectionSession:
        session = DetectionSession(
            session_type=session_type.upper(), video_name=video_name, user_id=user_id
        )
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
    def get_sessions(db: Session, user_id: Optional[int] = None) -> List[DetectionSession]:
        query = db.query(DetectionSession)
        if user_id is not None:
            query = query.filter(DetectionSession.user_id == user_id)
        return query.order_by(DetectionSession.started_at.desc()).all()

    @staticmethod
    def get_session_by_id(
        db: Session, session_id: int, user_id: Optional[int] = None
    ) -> Optional[DetectionSession]:
        query = db.query(DetectionSession).filter(DetectionSession.id == session_id)
        if user_id is not None:
            query = query.filter(DetectionSession.user_id == user_id)
        return query.first()

    @staticmethod
    def get_statistics(db: Session, user_id: Optional[int] = None) -> Dict[str, object]:
        base_session_q = db.query(DetectionSession)
        if user_id is not None:
            base_session_q = base_session_q.filter(DetectionSession.user_id == user_id)

        total_sessions = base_session_q.count()
        total_video_sessions = base_session_q.filter(DetectionSession.session_type == "VIDEO").count()
        total_camera_sessions = base_session_q.filter(DetectionSession.session_type == "CAMERA").count()

        obj_q = db.query(func.coalesce(func.sum(DetectionSession.total_objects), 0))
        if user_id is not None:
            obj_q = obj_q.filter(DetectionSession.user_id == user_id)
        total_objects = obj_q.scalar() or 0

        res_q = db.query(DetectionResult).join(
            DetectionSession, DetectionResult.session_id == DetectionSession.id
        )
        if user_id is not None:
            res_q = res_q.filter(DetectionSession.user_id == user_id)

        most_common = (
            res_q.with_entities(DetectionResult.class_name, func.count(DetectionResult.id).label("count"))
            .group_by(DetectionResult.class_name)
            .order_by(func.count(DetectionResult.id).desc())
            .first()
        )

        avg_conf_q = db.query(func.avg(DetectionResult.confidence)).join(
            DetectionSession, DetectionResult.session_id == DetectionSession.id
        )
        if user_id is not None:
            avg_conf_q = avg_conf_q.filter(DetectionSession.user_id == user_id)
        avg_confidence = avg_conf_q.scalar() or 0.0

        classes = (
            res_q.with_entities(DetectionResult.class_name, func.count(DetectionResult.id).label("count"))
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
    def get_recent_sessions(
        db: Session, user_id: Optional[int] = None, limit: int = 10
    ) -> List[DetectionSession]:
        query = db.query(DetectionSession)
        if user_id is not None:
            query = query.filter(DetectionSession.user_id == user_id)
        return query.order_by(DetectionSession.started_at.desc()).limit(limit).all()
