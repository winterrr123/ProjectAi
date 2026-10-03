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
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully")
    except OperationalError as exc:
        logger.warning("Database not available; continuing without DB initialization: %s", exc)


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
