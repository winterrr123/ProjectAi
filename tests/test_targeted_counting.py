import numpy as np
import pytest
from app.services.video_service import VideoProcessingService, is_matching_target


def test_is_matching_target():
    # Can matching
    assert is_matching_target("Lon Bia Saigon", "can") is True
    assert is_matching_target("beverage can", "can") is True
    assert is_matching_target("Chai Nước", "can") is False
    assert is_matching_target("Người", "can") is False
    assert is_matching_target("Ghế", "can") is False

    # Bottle matching
    assert is_matching_target("Chai Nước", "bottle") is True
    assert is_matching_target("plastic bottle", "bottle") is True
    assert is_matching_target("Lon Bia", "bottle") is False

    # Box matching
    assert is_matching_target("Thùng / Hộp Sản Phẩm", "box") is True
    assert is_matching_target("Gói Cà Phê", "box") is True
    assert is_matching_target("Người", "box") is False

    # All / None
    assert is_matching_target("Bất kỳ", None) is True
    assert is_matching_target("Bất kỳ", "all") is True


def test_targeted_filtering_in_process_frame(monkeypatch):
    service = VideoProcessingService()
    service.reset()

    # Mock raw detections from yolo
    mock_raw_dets = [
        {"class_name": "Lon Coca", "raw_name": "can", "x1": 50, "y1": 50, "x2": 150, "y2": 150, "confidence": 0.9, "tracking_id": 101},
        {"class_name": "Người", "raw_name": "person", "x1": 200, "y1": 50, "x2": 400, "y2": 450, "confidence": 0.92, "tracking_id": 102},
        {"class_name": "Chai Nước", "raw_name": "bottle", "x1": 420, "y1": 100, "x2": 480, "y2": 300, "confidence": 0.88, "tracking_id": 103},
        {"class_name": "Ghế", "raw_name": "chair", "x1": 500, "y1": 200, "x2": 600, "y2": 400, "confidence": 0.85, "tracking_id": 104},
    ]

    monkeypatch.setattr(service.yolo_service, "track_infer", lambda frame, conf: [d.copy() for d in mock_raw_dets])

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Target "can" with custom target_name
    res = service.process_frame(
        dummy_frame,
        frame_number=1,
        target_class="can",
        target_name="Lon Coca-Cola 330ml",
    )

    dets = res["detections"]
    assert len(dets) == 1
    assert dets[0]["class_name"] == "Lon Coca-Cola 330ml"
    assert dets[0]["x1"] == 50.0

    # 2. Target "bottle"
    service.reset()
    res_bottle = service.process_frame(
        dummy_frame,
        frame_number=1,
        target_class="bottle",
        target_name="Chai Nước Suối 500ml",
    )
    dets_bottle = res_bottle["detections"]
    assert len(dets_bottle) == 1
    assert dets_bottle[0]["class_name"] == "Chai Nước Suối 500ml"
    assert dets_bottle[0]["x1"] == 420.0

    # 3. No target specified (legacy mode)
    service.reset()
    res_legacy = service.process_frame(
        dummy_frame,
        frame_number=1,
        target_class=None,
    )
    # Returns all valid boxes labeled as "Sản phẩm"
    assert len(res_legacy["detections"]) >= 2
    for d in res_legacy["detections"]:
        assert d["class_name"] == "Sản phẩm"
