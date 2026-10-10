from app.config import settings


def test_auth_settings_exist():
    assert hasattr(settings, "GOOGLE_CLIENT_ID")
    assert hasattr(settings, "JWT_SECRET_KEY")
    assert hasattr(settings, "JWT_ALGORITHM")
    assert hasattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES")
    assert settings.JWT_ALGORITHM == "HS256"
    assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0
