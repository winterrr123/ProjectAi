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
