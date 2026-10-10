from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings
from app.utils.helpers import logger


class Base(DeclarativeBase):
    pass


# Configure engine options based on database type (SQLite vs MySQL)
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}
engine_kwargs = {"echo": False, "future": True}
if not is_sqlite:
    engine_kwargs["pool_pre_ping"] = True

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    **engine_kwargs,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def init_db() -> None:
    try:
        from app.models.detection import DetectionResult, DetectionSession, ProductCount  # noqa: F401
        from app.models.product import Product  # noqa: F401
        from app.models.user import User  # noqa: F401
        from sqlalchemy import inspect, text

        Base.metadata.create_all(bind=engine)

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Ensure default user exists if users table is created
        if "users" in tables:
            with engine.begin() as conn:
                res = conn.execute(text("SELECT id FROM users LIMIT 1")).first()
                if not res:
                    conn.execute(
                        text(
                            "INSERT INTO users (google_id, email, full_name, created_at) "
                            "VALUES (:gid, :email, :name, CURRENT_TIMESTAMP)"
                        ),
                        {"gid": "system_default", "email": "system@local.host", "name": "Default User"},
                    )

        # Migration: if detection_sessions exists without user_id, add it
        if "detection_sessions" in tables:
            columns = [c["name"] for c in inspector.get_columns("detection_sessions")]
            if "user_id" not in columns:
                logger.info("Migrating detection_sessions: adding user_id column")
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE detection_sessions ADD COLUMN user_id INTEGER DEFAULT 1"))

        logger.info("Database schema initialized successfully")
    except OperationalError as exc:
        logger.warning("Database not available; continuing without DB initialization: %s", exc)
    except Exception as exc:
        logger.exception("Error during database schema migration: %s", exc)


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
