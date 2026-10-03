import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings


ensure_log_dir = None


def ensure_log_dir() -> None:
    log_dir = settings.PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)


ensure_log_dir()

logger = logging.getLogger("product_detection")
logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
if not logger.handlers:
    handler = logging.FileHandler(settings.PROJECT_ROOT / "logs" / "app.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def safe_filename(filename: str) -> str:
    return str(uuid.uuid4()) + Path(filename).suffix.lower()


def is_valid_video(file_path: str) -> bool:
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
    return Path(file_path).suffix.lower() in allowed_ext


def compute_stats(counts: Dict[str, int]) -> Dict[str, Any]:
    total = sum(counts.values())
    by_class = {key: int(value) for key, value in counts.items()}
    return {"total": total, "by_class": by_class}


def format_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def get_video_output_path(input_name: str) -> str:
    stem = Path(input_name).stem
    return str(settings.OUTPUT_FOLDER / f"{stem}_processed.mp4")
