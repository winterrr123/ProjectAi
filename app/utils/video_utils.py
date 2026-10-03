from __future__ import annotations

from pathlib import Path


def allowed_video_extensions() -> set[str]:
    return {".mp4", ".avi", ".mov", ".mkv"}


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
