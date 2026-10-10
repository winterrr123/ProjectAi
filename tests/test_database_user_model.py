from app.models.detection import DetectionSession
from app.models.user import User


def test_user_model_attributes():
    user = User(
        google_id="google_12345",
        email="test@example.com",
        full_name="Test User",
        avatar_url="https://example.com/avatar.png",
    )
    assert user.google_id == "google_12345"
    assert user.email == "test@example.com"
    assert user.full_name == "Test User"
    assert user.avatar_url == "https://example.com/avatar.png"


def test_detection_session_has_user_id():
    session = DetectionSession(session_type="CAMERA", user_id=1)
    assert hasattr(session, "user_id")
    assert session.user_id == 1


def test_init_db_creates_tables():
    from app.database.mysql import init_db, engine
    from sqlalchemy import inspect
    init_db()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "users" in tables
    assert "detection_sessions" in tables
    columns = [c["name"] for c in inspector.get_columns("detection_sessions")]
    assert "user_id" in columns
