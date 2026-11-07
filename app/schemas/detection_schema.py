from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    session_type: str = Field(default="VIDEO")
    video_name: Optional[str] = None
    target_class: Optional[str] = None
    target_name: Optional[str] = None


class DetectionResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[int] = None
    output_video: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class CameraSessionPayload(BaseModel):
    action: str = Field(..., pattern="^(start|stop)$")


class FrameDetectionPayload(BaseModel):
    image: str = Field(..., description="Base64 encoded image frame")
    frame_number: int = Field(default=1)
    reset: bool = Field(default=False)
    session_id: Optional[str] = None
    source: Optional[str] = Field(default="upload", description="'upload' or 'camera'")
    target_class: Optional[str] = None
    target_name: Optional[str] = None


class SaveLiveSessionPayload(BaseModel):
    session_type: str = Field(default="VIDEO")
    video_name: str = Field(default="Live Video Session")
    total_objects: int = Field(default=0)
    by_class: Dict[str, int] = Field(default_factory=dict)


class ExportExcelRequest(BaseModel):
    session_id: Optional[int] = None
    video_name: Optional[str] = "Live_Counting_Session.mp4"
    total_count: Optional[int] = 0
    by_class: Optional[Dict[str, int]] = Field(default_factory=dict)
    target_name: Optional[str] = None
    detections: Optional[list] = None
