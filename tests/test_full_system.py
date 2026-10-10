from fastapi.testclient import TestClient

from app.database.crud import DetectionCRUD, SessionCRUD, UserCRUD
from app.database.mysql import SessionLocal, init_db
from app.main import app
from app.utils.auth import create_access_token

client = TestClient(app, follow_redirects=False)


def test_health_check():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["success"] is True


import uuid

def test_complete_multi_user_isolation_flow():
    init_db()
    db = SessionLocal()
    uid = uuid.uuid4().hex[:8]
    try:
        # 1. Create two separate Google users
        user_alpha = UserCRUD.get_or_create_google_user(
            db,
            google_id=f"g_alpha_{uid}",
            email=f"alpha_{uid}@test.org",
            full_name="Alpha Engineer",
            avatar_url="https://test.org/alpha.png",
        )
        user_beta = UserCRUD.get_or_create_google_user(
            db,
            google_id=f"g_beta_{uid}",
            email=f"beta_{uid}@test.org",
            full_name="Beta Engineer",
            avatar_url="https://test.org/beta.png",
        )

        token_alpha = create_access_token({"sub": str(user_alpha.id), "email": user_alpha.email})
        token_beta = create_access_token({"sub": str(user_beta.id), "email": user_beta.email})

        # 2. User Alpha saves a live camera session
        res_save_alpha = client.post(
            "/api/video/save-session",
            json={
                "session_type": "CAMERA",
                "video_name": "line_1_camera",
                "total_objects": 25,
                "by_class": {"chai nước": 15, "hộp sữa": 10},
            },
            cookies={"access_token": token_alpha},
        )
        assert res_save_alpha.status_code == 200
        session_alpha_id = res_save_alpha.json()["session_id"]

        # 3. User Beta saves a video session
        res_save_beta = client.post(
            "/api/video/save-session",
            json={
                "session_type": "VIDEO",
                "video_name": "warehouse_feed.mp4",
                "total_objects": 40,
                "by_class": {"thùng hộp": 40},
            },
            cookies={"access_token": token_beta},
        )
        assert res_save_beta.status_code == 200
        session_beta_id = res_save_beta.json()["session_id"]

        # 4. Check User Alpha's view:
        alpha_sessions_res = client.get("/api/sessions", cookies={"access_token": token_alpha})
        assert alpha_sessions_res.status_code == 200
        alpha_ids = [s["id"] for s in alpha_sessions_res.json()["sessions"]]
        assert session_alpha_id in alpha_ids
        assert session_beta_id not in alpha_ids

        alpha_stats_res = client.get("/api/statistics", cookies={"access_token": token_alpha})
        assert alpha_stats_res.status_code == 200
        alpha_stats = alpha_stats_res.json()["statistics"]
        assert alpha_stats["total_objects"] == 25
        assert "thùng hộp" not in alpha_stats["class_counts"]

        # 5. Check User Beta's view:
        beta_sessions_res = client.get("/api/sessions", cookies={"access_token": token_beta})
        assert beta_sessions_res.status_code == 200
        beta_ids = [s["id"] for s in beta_sessions_res.json()["sessions"]]
        assert session_beta_id in beta_ids
        assert session_alpha_id not in beta_ids

        beta_stats_res = client.get("/api/statistics", cookies={"access_token": token_beta})
        assert beta_stats_res.status_code == 200
        beta_stats = beta_stats_res.json()["statistics"]
        assert beta_stats["total_objects"] == 40
        assert "chai nước" not in beta_stats["class_counts"]

        # 6. User Alpha tries to get User Beta's session by ID -> 404
        hijack_res = client.get(f"/api/sessions/{session_beta_id}", cookies={"access_token": token_alpha})
        assert hijack_res.status_code == 404

        # 7. Unauthenticated access to /api/sessions -> 401
        unauth_api = client.get("/api/sessions")
        assert unauth_api.status_code == 401

        # 8. Unauthenticated access to pages -> 303 Redirect to /login
        unauth_page = client.get("/camera")
        assert unauth_page.status_code in (302, 303, 307)
        assert "/login" in unauth_page.headers.get("location", "")

    finally:
        db.close()
