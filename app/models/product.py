from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from app.database.mysql import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False, unique=True)
    class_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
