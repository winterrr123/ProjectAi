from fastapi.testclient import TestClient

from app.database.crud import SessionCRUD, UserCRUD
from app.database.mysql import SessionLocal, init_db
from app.main import app
from app.utils.auth import create_access_token

client = TestClient(app)


def test_user_data_isolation():
    init_db()
    db = SessionLocal()
    try:
        user_a = UserCRUD.get_or_create_google_user(db, "g_iso_a", "a_iso@test.com", "User A", None)
        user_b = UserCRUD.get_or_create_google_user(db, "g_iso_b", "b_iso@test.com", "User B", None)

        session_a = SessionCRUD.create_session(db, "CAMERA", "live_stream_a", user_id=user_a.id)
        SessionCRUD.update_session_total(db, session_a.id, 15)

        session_b = SessionCRUD.create_session(db, "VIDEO", "test_b.mp4", user_id=user_b.id)
        SessionCRUD.update_session_total(db, session_b.id, 8)

        # 1. User A should only see Session A in CRUD
        user_a_sessions = SessionCRUD.get_sessions(db, user_id=user_a.id)
        assert len(user_a_sessions) >= 1
        assert all(s.user_id == user_a.id for s in user_a_sessions)
        assert not any(s.id == session_b.id for s in user_a_sessions)

        # 2. User A cannot access Session B by ID in CRUD
        assert SessionCRUD.get_session_by_id(db, session_b.id, user_id=user_a.id) is None
        assert SessionCRUD.get_session_by_id(db, session_a.id, user_id=user_a.id) is not None

        # 3. User A stats only count User A objects
        stats_a = SessionCRUD.get_statistics(db, user_id=user_a.id)
        assert stats_a["total_objects"] >= 15
        assert stats_a["total_camera_sessions"] >= 1

        # 4. API isolation test: /api/sessions with token A
        token_a = create_access_token({"sub": str(user_a.id), "email": user_a.email})
        res_a = client.get("/api/sessions", cookies={"access_token": token_a})
        assert res_a.status_code == 200
        sessions_data = res_a.json().get("sessions", [])
        assert all(s["id"] != session_b.id for s in sessions_data)
        assert any(s["id"] == session_a.id for s in sessions_data)

        # 5. User A querying /api/sessions/{session_b.id} must be 404
        res_forbidden = client.get(f"/api/sessions/{session_b.id}", cookies={"access_token": token_a})
        assert res_forbidden.status_code == 404
    finally:
        db.close()
