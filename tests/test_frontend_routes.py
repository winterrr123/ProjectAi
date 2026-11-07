from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_login_page_renders():
    response = client.get("/login")
    assert response.status_code == 200
    assert "Đăng nhập" in response.text
    assert "accounts.google.com/gsi/client" in response.text
    assert "g_id_onload" in response.text


def test_auth_js_is_served():
    response = client.get("/static/js/auth.js")
    assert response.status_code == 200
    assert "initAuthNavbar" in response.text


def test_camera_page_renders_with_sample_station():
    from app.database.crud import UserCRUD
    from app.database.mysql import SessionLocal, init_db
    from app.utils.auth import create_access_token

    init_db()
    db = SessionLocal()
    try:
        user = UserCRUD.get_or_create_google_user(
            db,
            google_id="g_test_cam_render",
            email="cam_render@example.com",
            full_name="Cam Tester",
            avatar_url="",
        )
        token = create_access_token({"sub": str(user.id), "email": user.email})
        response = client.get("/camera", cookies={"access_token": token})
        assert response.status_code == 200
        assert "Trạm Nhận Diện Mẫu Sản Phẩm Đơn Vị" in response.text
        assert 'id="cameraViewfinder"' in response.text
        assert 'id="identifySampleBtn"' in response.text
        assert 'id="openLibraryBtn"' in response.text
        assert 'id="sampleLibraryModal"' in response.text
    finally:
        db.close()


def test_upload_page_renders_with_target_studio():
    from app.database.crud import UserCRUD
    from app.database.mysql import SessionLocal, init_db
    from app.utils.auth import create_access_token

    init_db()
    db = SessionLocal()
    try:
        user = UserCRUD.get_or_create_google_user(
            db,
            google_id="g_test_up_render",
            email="up_render@example.com",
            full_name="Upload Tester",
            avatar_url="",
        )
        token = create_access_token({"sub": str(user.id), "email": user.email})
        response = client.get("/upload", cookies={"access_token": token})
        assert response.status_code == 200
        assert "Studio Tải & Đếm Video" in response.text
        assert 'id="activeTargetBannerContainer"' in response.text
        assert 'id="targetSwitcherModal"' in response.text
    finally:
        db.close()


