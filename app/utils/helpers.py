import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def cleanup_old_files(days: int = 7) -> Dict[str, Any]:
    """
    Quét thư mục uploads/ (bao gồm thư mục con như samples/) và outputs/,
    xóa các file cũ hơn số ngày quy định (ngoại trừ .gitkeep).
    """
    cutoff = time.time() - (float(days) * 86400)
    deleted_files: List[str] = []
    errors: List[str] = []

    target_dirs = [Path(settings.UPLOAD_FOLDER), Path(settings.OUTPUT_FOLDER)]

    for directory in target_dirs:
        if not directory.exists():
            continue
        for file_path in directory.rglob("*"):
            try:
                if file_path.is_file():
                    if file_path.name == ".gitkeep":
                        continue
                    if file_path.stat().st_mtime < cutoff:
                        file_path.unlink()
                        deleted_files.append(str(file_path))
            except Exception as exc:
                logger.warning("Could not delete old file %s: %s", file_path, exc)
                errors.append(f"{file_path}: {exc}")

    logger.info("cleanup_old_files(days=%s) removed %d files", days, len(deleted_files))
    return {
        "deleted_count": len(deleted_files),
        "deleted_files": deleted_files,
        "errors": errors,
    }
