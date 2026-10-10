from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database.mysql import Base


class DetectionSession(Base):
    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_type = Column(String(50), nullable=False)
    video_name = Column(String(255), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    total_objects = Column(Integer, default=0, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    user = relationship("User", back_populates="sessions")
    detections = relationship("DetectionResult", back_populates="session")
    counts = relationship("ProductCount", back_populates="session")


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=False)
    tracking_id = Column(Integer, nullable=True)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)

    session = relationship("DetectionSession", back_populates="detections")


class ProductCount(Base):
    __tablename__ = "product_counts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=False)
    class_name = Column(String(255), nullable=False)
    count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("DetectionSession", back_populates="counts")
