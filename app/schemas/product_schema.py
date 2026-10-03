from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ProductCreate(BaseModel):
    product_name: str = Field(..., min_length=1)
    class_id: int = Field(..., ge=0)


class ProductResponse(ProductCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
