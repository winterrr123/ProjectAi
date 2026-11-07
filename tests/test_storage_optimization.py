import base64
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database.crud import DetectionCRUD, SampleCRUD, SessionCRUD, UserCRUD
from app.database.mysql import SessionLocal
from app.main import app
from app.models.detection import DetectionResult, DetectionSession, ProductCount
from app.services.vision_api_service import GeminiVisionService
from app.utils.auth import create_access_token
from app.utils.helpers import cleanup_old_files


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def auth_user_a(db_session):
    user = UserCRUD.get_or_create_google_user(
        db_session,
        google_id="storage_opt_user_a",
        email="storage_a@example.com",
        full_name="Storage User A",
    )
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return user, token


@pytest.fixture
def auth_user_b(db_session):
    user = UserCRUD.get_or_create_google_user(
        db_session,
        google_id="storage_opt_user_b",
        email="storage_b@example.com",
        full_name="Storage User B",
    )
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return user, token


# ------------------------------------------------------------------------------
# 1. Bulk Insert for Detection Results
# ------------------------------------------------------------------------------
def test_save_detection_results_bulk_and_chunking(db_session, auth_user_a):
    user, _ = auth_user_a
    session = SessionCRUD.create_session(db_session, session_type="VIDEO", video_name="bulk_test.mp4", user_id=user.id)

    # Empty list handling
    DetectionCRUD.save_detection_results(db_session, session.id, [])
    count = db_session.query(DetectionResult).filter(DetectionResult.session_id == session.id).count()
    assert count == 0

    # Large batch: 1250 items to test chunking (>1000)
    items = []
    for i in range(1250):
        items.append({
            "tracking_id": i + 1,
            "class_id": 1,
            "class_name": "Can",
            "confidence": 0.85,
            "x1": 10.0 + i,
            "y1": 20.0,
            "x2": 50.0 + i,
            "y2": 80.0,
            "frame_number": i // 10,
            "timestamp": float(i) * 0.033,
        })

    DetectionCRUD.save_detection_results(db_session, session.id, items)

    saved_count = db_session.query(DetectionResult).filter(DetectionResult.session_id == session.id).count()
    assert saved_count == 1250

    # Verify first and last records
    first_res = (
        db_session.query(DetectionResult)
        .filter(DetectionResult.session_id == session.id)
        .order_by(DetectionResult.id.asc())
        .first()
    )
    assert first_res.tracking_id == 1
    assert first_res.class_name == "Can"
    assert round(first_res.confidence, 2) == 0.85

    # Cleanup
    SessionCRUD.delete_session(db_session, session.id, user.id)


# ------------------------------------------------------------------------------
# 2. Sample Image Storage Optimization & Backward Compatibility
# ------------------------------------------------------------------------------
def test_sample_image_storage_and_endpoint(client, auth_user_a, monkeypatch):
    user, token = auth_user_a

    mock_gemini_res = {
        "success": True,
        "data": {
            "product_name": "Storage Test Drink 500ml",
            "brand": "OptiBrand",
            "category": "Đồ uống",
            "visual_class": "bottle",
            "specs": "500ml",
        },
    }
    monkeypatch.setattr(
        GeminiVisionService, "identify_sample_product", lambda self, image_bytes, api_key=None: mock_gemini_res
    )

    fake_bytes = b"opti_image_content_test_123"
    b64_payload = base64.b64encode(fake_bytes).decode("utf-8")

    res = client.post(
        "/api/vision/identify-sample",
        json={"image": f"data:image/jpeg;base64,{b64_payload}"},
        cookies={"access_token": token},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["success"] is True
    sample = data["sample"]

    # Verify sample_image_url is stored as URL, not Base64
    image_url = sample["sample_image_url"]
    assert image_url.startswith("/api/video/sample-image/")
    filename = image_url.split("/")[-1]
    assert filename.endswith(".jpg")

    # Verify physical file exists on disk
    file_on_disk = settings.UPLOAD_FOLDER / "samples" / filename
    assert file_on_disk.exists()
    assert file_on_disk.read_bytes() == fake_bytes

    # Verify GET /api/video/sample-image/{filename}
    img_res = client.get(image_url)
    assert img_res.status_code == 200
    assert img_res.content == fake_bytes

    # Verify static mount /uploads/samples/{filename}
    mount_res = client.get(f"/uploads/samples/{filename}")
    assert mount_res.status_code == 200
    assert mount_res.content == fake_bytes

    # Cleanup test image file
    if file_on_disk.exists():
        file_on_disk.unlink()


def test_sample_backward_compatibility_with_base64(client, db_session, auth_user_a):
    user, token = auth_user_a

    # Legacy sample with Base64 data URL directly in DB
    legacy_b64_url = "data:image/jpeg;base64,LEGACY_BASE64_STRING_FOR_TESTING"
    sample = SampleCRUD.create_sample(
        db=db_session,
        user_id=user.id,
        product_name="Legacy Product",
        brand="Legacy Brand",
        visual_class="can",
        sample_image_url=legacy_b64_url,
    )

    res = client.get("/api/vision/samples", cookies={"access_token": token})
    assert res.status_code == 200
    samples = res.json()["samples"]
    found = next((s for s in samples if s["id"] == sample.id), None)
    assert found is not None
    assert found["sample_image_url"] == legacy_b64_url

    # Clean up
    SampleCRUD.delete_sample(db_session, sample.id, user.id)


# ------------------------------------------------------------------------------
# 3. DELETE /api/sessions/{session_id} with file cleanup & security checks
# ------------------------------------------------------------------------------
def test_delete_session_with_file_cleanup(client, db_session, auth_user_a, auth_user_b):
    user_a, token_a = auth_user_a
    user_b, token_b = auth_user_b

    # 1. Create a session for User A
    session = SessionCRUD.create_session(db_session, session_type="VIDEO", video_name="delete_test.mp4", user_id=user_a.id)
    session_id = session.id
    DetectionCRUD.save_detection_results(
        db_session,
        session_id,
        [{"class_name": "Can", "confidence": 0.9, "frame_number": 1}],
    )
    DetectionCRUD.save_product_counts(db_session, session_id, {"Can": 1})

    # Create dummy output files on disk
    dummy_video_path = settings.OUTPUT_FOLDER / f"processed_{session_id}_delete_test.mp4"
    dummy_excel_path = settings.OUTPUT_FOLDER / f"report_session_{session_id}_delete_test.xlsx"
    dummy_video_path.write_bytes(b"dummy video result content")
    dummy_excel_path.write_bytes(b"dummy excel report content")

    assert dummy_video_path.exists()
    assert dummy_excel_path.exists()

    # 2. User B tries to delete User A's session -> 404 (Security check)
    del_res_b = client.delete(f"/api/sessions/{session_id}", cookies={"access_token": token_b})
    assert del_res_b.status_code == 404

    # 3. Unauthenticated request -> 401
    unauth_res = client.delete(f"/api/sessions/{session_id}")
    assert unauth_res.status_code == 401

    # 4. User A deletes their own session -> 200
    del_res_a = client.delete(f"/api/sessions/{session_id}", cookies={"access_token": token_a})
    assert del_res_a.status_code == 200
    data = del_res_a.json()
    assert data["success"] is True
    assert dummy_video_path.name in data["deleted_files"]
    assert dummy_excel_path.name in data["deleted_files"]

    # 5. Verify records removed from database
    verify_db = SessionLocal()
    try:
        assert SessionCRUD.get_session_by_id(verify_db, session_id) is None
        assert verify_db.query(DetectionResult).filter(DetectionResult.session_id == session_id).count() == 0
        assert verify_db.query(ProductCount).filter(ProductCount.session_id == session_id).count() == 0
    finally:
        verify_db.close()

    # 6. Verify files deleted from disk
    assert not dummy_video_path.exists()
    assert not dummy_excel_path.exists()

    # 7. Trying to delete again -> 404
    del_res_again = client.delete(f"/api/sessions/{session_id}", cookies={"access_token": token_a})
    assert del_res_again.status_code == 404


# ------------------------------------------------------------------------------
# 4. Storage Auto-Cleanup Utility
# ------------------------------------------------------------------------------
def test_cleanup_old_files():
    # Setup test directory files in outputs/
    old_file = settings.OUTPUT_FOLDER / "test_old_file_to_cleanup.mp4"
    recent_file = settings.OUTPUT_FOLDER / "test_recent_file_to_keep.mp4"
    gitkeep_file = settings.OUTPUT_FOLDER / ".gitkeep"

    old_file.write_bytes(b"old content")
    recent_file.write_bytes(b"recent content")

    # Ensure gitkeep exists
    if not gitkeep_file.exists():
        gitkeep_file.write_bytes(b"")

    try:
        # Set old_file modification time to 10 days ago (10 * 86400 seconds)
        ten_days_ago = time.time() - (10 * 86400)
        os.utime(str(old_file), (ten_days_ago, ten_days_ago))

        # Recent file is right now
        now = time.time()
        os.utime(str(recent_file), (now, now))

        # Run cleanup for files older than 7 days
        result = cleanup_old_files(days=7)

        # Assertions
        assert not old_file.exists(), "Old file should have been deleted"
        assert recent_file.exists(), "Recent file should have been kept"
        assert gitkeep_file.exists(), ".gitkeep file should never be deleted"
        assert str(old_file) in result["deleted_files"]
    finally:
        # Cleanup any remaining test files
        if old_file.exists():
            old_file.unlink()
        if recent_file.exists():
            recent_file.unlink()
