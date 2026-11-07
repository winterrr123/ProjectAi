from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database.mysql import Base


class SampleProduct(Base):
    __tablename__ = "sample_products"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    brand = Column(String(255), nullable=True)
    category = Column(String(255), nullable=True)
    visual_class = Column(String(100), nullable=False, default="product")
    sample_image_url = Column(Text, nullable=True)
    specs = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="samples")
