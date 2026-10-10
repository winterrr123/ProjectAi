from fastapi.testclient import TestClient

from app.database.crud import UserCRUD
from app.database.mysql import SessionLocal, init_db
from app.main import app
from app.utils.auth import create_access_token

client = TestClient(app, follow_redirects=False)


def test_unauthenticated_api_me_returns_401():
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_unauthenticated_history_page_redirects():
    response = client.get("/history")
    assert response.status_code in (302, 303, 307)
    assert "/login" in response.headers.get("location", "")


def test_authenticated_api_me():
    init_db()
    db = SessionLocal()
    try:
        user = UserCRUD.get_or_create_google_user(
            db,
            google_id="g_test_auth_routes",
            email="auth_routes@example.com",
            full_name="Auth Route Tester",
            avatar_url="https://example.com/pic.png",
        )
        token = create_access_token({"sub": str(user.id), "email": user.email})
        response = client.get("/api/auth/me", cookies={"access_token": token})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user"]["email"] == "auth_routes@example.com"
    finally:
        db.close()


def test_logout():
    response = client.post("/api/auth/logout")
    assert response.status_code == 200
    assert response.json()["success"] is True
