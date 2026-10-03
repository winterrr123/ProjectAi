import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_ROOT = BASE_DIR
    APP_DIR = BASE_DIR / "app"
    FRONTEND_DIR = BASE_DIR / "frontend"
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.pt"))
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'product_detection.db'}",
    )
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.20"))
    DEVICE = os.getenv("DEVICE", "cpu")
    UPLOAD_FOLDER = BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads")
    OUTPUT_FOLDER = BASE_DIR / os.getenv("OUTPUT_FOLDER", "outputs")
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "524288000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    VISION_ENGINE = os.getenv("VISION_ENGINE", "standard")  # 'standard', 'yolo_world', 'hybrid'
    YOLO_WORLD_CLASSES = os.getenv(
        "YOLO_WORLD_CLASSES",
        "gói cà phê, chai nước, hộp sữa, lon nước ngọt, bánh kẹo, gói snack, điện thoại, máy tính, sách vở, balo, túi xách, thùng hộp"
    )

    def __post_init__(self):
        self.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


settings = Settings()
