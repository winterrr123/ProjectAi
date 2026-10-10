from datetime import timedelta

from app.database.crud import UserCRUD
from app.database.mysql import SessionLocal, init_db
from app.utils.auth import create_access_token, decode_access_token


def test_jwt_create_and_decode():
    token = create_access_token({"sub": "123", "email": "user@test.com"}, expires_delta=timedelta(minutes=5))
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "123"
    assert payload["email"] == "user@test.com"


def test_jwt_expired():
    token = create_access_token({"sub": "123"}, expires_delta=timedelta(seconds=-10))
    payload = decode_access_token(token)
    assert payload is None


def test_user_crud_get_or_create():
    init_db()
    db = SessionLocal()
    try:
        user1 = UserCRUD.get_or_create_google_user(
            db,
            google_id="g_test_101",
            email="test101@example.com",
            full_name="Tester 101",
            avatar_url="https://example.com/pic.jpg",
        )
        assert user1.id is not None
        assert user1.google_id == "g_test_101"

        # Query again, should return same user
        user2 = UserCRUD.get_or_create_google_user(
            db,
            google_id="g_test_101",
            email="test101@example.com",
            full_name="Tester 101 Updated",
            avatar_url="https://example.com/pic.jpg",
        )
        assert user2.id == user1.id

        # Query by id
        user_by_id = UserCRUD.get_by_id(db, user1.id)
        assert user_by_id is not None
        assert user_by_id.email == "test101@example.com"
    finally:
        db.close()
