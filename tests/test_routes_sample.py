import base64
import pytest
from fastapi.testclient import TestClient

from app.database.crud import SampleCRUD, UserCRUD
from app.database.mysql import SessionLocal, init_db
from app.main import app
from app.services.vision_api_service import GeminiVisionService, normalize_visual_class
from app.utils.auth import create_access_token

client = TestClient(app)


def test_normalize_visual_class():
    assert normalize_visual_class("Beverage Can") == "can"
    assert normalize_visual_class("lon coca") == "can"
    assert normalize_visual_class("plastic bottle") == "bottle"
    assert normalize_visual_class("chai nuoc") == "bottle"
    assert normalize_visual_class("paper cup") == "cup"
    assert normalize_visual_class("hop sua carton") == "box"
    assert normalize_visual_class("unknown_thing") == "unknown_thing"
    assert normalize_visual_class("unknown thing!@#") == "unknownthing"
    assert normalize_visual_class(None) == "product"


def test_identify_sample_api(monkeypatch):
    init_db()
    db = SessionLocal()
    try:
        user = UserCRUD.get_or_create_google_user(
            db, "g_sample_test", "sample_test@test.com", "Sample Tester", None
        )
        token = create_access_token({"sub": str(user.id), "email": user.email})

        # Mock GeminiVisionService.identify_sample_product
        mock_result = {
            "success": True,
            "data": {
                "product_name": "Lon Bia Saigon Special 330ml",
                "brand": "Sabeco",
                "category": "Bia",
                "visual_class": "can",
                "specs": "330ml",
                "confidence": 0.98,
            },
        }
        monkeypatch.setattr(
            GeminiVisionService, "identify_sample_product", lambda self, image_bytes, api_key=None: mock_result
        )

        dummy_img = base64.b64encode(b"fake_jpeg_image_content").decode("utf-8")
        payload = {"image": f"data:image/jpeg;base64,{dummy_img}"}

        response = client.post(
            "/api/vision/identify-sample",
            json=payload,
            cookies={"access_token": token},
        )
        assert response.status_code == 200, response.text
        res_data = response.json()
        assert res_data["success"] is True
        sample = res_data["sample"]
        assert sample["product_name"] == "Lon Bia Saigon Special 330ml"
        assert sample["visual_class"] == "can"
        assert sample["brand"] == "Sabeco"
        assert sample["user_id"] == user.id

        # Verify persisted in database
        db.commit()
        saved = SampleCRUD.get_sample_by_id(db, sample["id"], user_id=user.id)
        assert saved is not None
        assert saved.product_name == "Lon Bia Saigon Special 330ml"
    finally:
        db.close()


def test_get_and_delete_samples_api():
    init_db()
    db = SessionLocal()
    try:
        user_a = UserCRUD.get_or_create_google_user(
            db, "g_sample_a", "sample_a@test.com", "User A", None
        )
        user_b = UserCRUD.get_or_create_google_user(
            db, "g_sample_b", "sample_b@test.com", "User B", None
        )
        token_a = create_access_token({"sub": str(user_a.id), "email": user_a.email})
        token_b = create_access_token({"sub": str(user_b.id), "email": user_b.email})

        s_a1 = SampleCRUD.create_sample(db, user_a.id, "Product A1", "Brand A", visual_class="can")
        s_a2 = SampleCRUD.create_sample(db, user_a.id, "Product A2", "Brand A", visual_class="bottle")
        s_b1 = SampleCRUD.create_sample(db, user_b.id, "Product B1", "Brand B", visual_class="box")

        # GET /api/vision/samples for User A
        res_a = client.get("/api/vision/samples", cookies={"access_token": token_a})
        assert res_a.status_code == 200
        samples_a = res_a.json()["samples"]
        assert len(samples_a) >= 2
        ids_a = [s["id"] for s in samples_a]
        assert s_a1.id in ids_a
        assert s_a2.id in ids_a
        assert s_b1.id not in ids_a

        # User B tries to delete User A's sample -> should fail (404)
        del_fail = client.delete(f"/api/vision/samples/{s_a1.id}", cookies={"access_token": token_b})
        assert del_fail.status_code == 404

        # User A deletes their own sample -> should succeed (200)
        del_ok = client.delete(f"/api/vision/samples/{s_a1.id}", cookies={"access_token": token_a})
        assert del_ok.status_code == 200
        assert del_ok.json()["success"] is True

        # Verify sample s_a1 is gone
        res_a_after = client.get("/api/vision/samples", cookies={"access_token": token_a})
        ids_after = [s["id"] for s in res_a_after.json()["samples"]]
        assert s_a1.id not in ids_after
    finally:
        db.close()


def test_identify_sample_without_gemini_key_uses_local_fallback():
    import numpy as np
    import cv2
    from app.config import settings
    init_db()
    db = SessionLocal()
    try:
        user = UserCRUD.get_or_create_google_user(
            db, "g_fallback_user", "fallback@test.com", "Fallback Tester", None
        )
        token = create_access_token({"sub": str(user.id), "email": user.email})

        # Ensure GEMINI_API_KEY is empty
        old_key = settings.GEMINI_API_KEY
        settings.GEMINI_API_KEY = ""

        # Create dummy valid JPEG bytes
        dummy_cv_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", dummy_cv_img)
        dummy_b64 = base64.b64encode(enc.tobytes()).decode("utf-8")

        response = client.post(
            "/api/vision/identify-sample",
            json={"image": f"data:image/jpeg;base64,{dummy_b64}"},
            cookies={"access_token": token},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert "sample" in data
        assert data["sample"]["visual_class"] in ["can", "bottle", "box", "cup", "product"]
        assert data["sample"]["user_id"] == user.id
    finally:
        settings.GEMINI_API_KEY = old_key
        db.close()

