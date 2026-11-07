import base64
import pytest
from fastapi.testclient import TestClient

from app.database.crud import UserCRUD
from app.database.mysql import SessionLocal, init_db
from app.main import app
from app.services.vision_api_service import GeminiVisionService
from app.utils.auth import create_access_token

client = TestClient(app)


def test_targeted_pipeline_e2e(monkeypatch, tmp_path):
    init_db()
    db = SessionLocal()
    try:
        user = UserCRUD.get_or_create_google_user(
            db, "g_e2e_user", "e2e_user@test.com", "E2E Tester", None
        )
        token = create_access_token({"sub": str(user.id), "email": user.email})

        # 1. Step 1: User photographs a sample product in Live Camera -> Gemini identifies it
        mock_gemini_sample = {
            "success": True,
            "data": {
                "product_name": "Lon Bia Heineken Silver 330ml",
                "brand": "Heineken",
                "category": "Bia",
                "visual_class": "can",
                "specs": "330ml",
                "confidence": 0.99,
            },
        }
        monkeypatch.setattr(
            GeminiVisionService,
            "identify_sample_product",
            lambda self, image_bytes, api_key=None: mock_gemini_sample,
        )

        dummy_img = base64.b64encode(b"fake_camera_frame_content").decode("utf-8")
        id_res = client.post(
            "/api/vision/identify-sample",
            json={"image": f"data:image/jpeg;base64,{dummy_img}"},
            cookies={"access_token": token},
        )
        assert id_res.status_code == 200, id_res.text
        sample_data = id_res.json()["sample"]
        assert sample_data["product_name"] == "Lon Bia Heineken Silver 330ml"
        assert sample_data["visual_class"] == "can"
        assert sample_data["user_id"] == user.id

        # 2. Step 2: Sample is in User's sample library (persists across sessions)
        lib_res = client.get("/api/vision/samples", cookies={"access_token": token})
        assert lib_res.status_code == 200
        samples = lib_res.json()["samples"]
        assert any(s["id"] == sample_data["id"] for s in samples)

        # 3. Step 3: Video processing uses target_class="can" and target_name="Lon Bia Heineken Silver 330ml"
        # Test HTTP detection frame with target filtering
        import cv2
        import numpy as np
        img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img_arr)
        fake_frame = base64.b64encode(buf.tobytes()).decode("utf-8")
        detect_res = client.post(
            "/api/detect/frame",
            json={
                "image": f"data:image/jpeg;base64,{fake_frame}",
                "frame_number": 1,
                "source": "video",
                "target_class": sample_data["visual_class"],
                "target_name": sample_data["product_name"],
            },
            cookies={"access_token": token},
        )
        assert detect_res.status_code == 200, detect_res.text
        det_data = detect_res.json()
        assert det_data["success"] is True

        # 4. Step 4: Deleting sample cleans it from library
        del_res = client.delete(
            f"/api/vision/samples/{sample_data['id']}",
            cookies={"access_token": token},
        )
        assert del_res.status_code == 200
    finally:
        db.close()
