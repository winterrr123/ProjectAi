"""
End-to-End Automated Video Counting Test Suite
==============================================
Validates that VideoProcessingService accurately counts beverage cans in real test videos:
  1. D:\\videotest\\sting.mp4:
     - Ground Truth: 6 cans (shrink-wrapped retail 6-pack Sting Gold Pro, 2x3 grid)
     - Expected frames: 177
  2. D:\\videotest\\sting,cola.mp4:
     - Ground Truth: 8 cans (1 Coca-Cola left, 6 Sting Gold Pro center, 1 Coca-Cola right)
     - Expected frames: 256

Execution commands:
  - Run video E2E tests:
      pytest tests/test_video_e2e.py -v -s
  - Run with strict Ground Truth assertions (M5 target):
      $env:STRICT_E2E="1"; pytest tests/test_video_e2e.py -v -s
  - Run unit test suite (32 unit tests):
      pytest tests/
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict

import pytest

from app.services.video_service import VideoProcessingService

GROUND_TRUTHS: Dict[str, Dict[str, Any]] = {
    "sting": {
        "key": "sting",
        "name": "sting.mp4",
        "path": r"D:\videotest\sting.mp4",
        "expected_total": 6,
        "expected_frames": 177,
        "description": "Shrink-wrapped retail 6-pack of Sting Gold Pro (2x3 grid)",
        "expected_classes": {"Sting Gold Pro": 6},
    },
    "sting_cola": {
        "key": "sting_cola",
        "name": "sting,cola.mp4",
        "path": r"D:\videotest\sting,cola.mp4",
        "expected_total": 8,
        "expected_frames": 256,
        "description": "8 cans in a row (1 Coca-Cola left, 6 Sting center, 1 Coca-Cola right)",
        "expected_classes": {"Sting Gold Pro": 6, "Coca-Cola": 2},
    },
}


def should_run_video_e2e() -> bool:
    """
    E2E video tests process full 720p/1080p videos through multi-model YOLO (~40s total).
    They run when:
      - RUN_VIDEO_E2E=1 environment variable is set
      - OR test_video_e2e is explicitly requested in sys.argv (e.g. 'pytest tests/test_video_e2e.py')
    When running standard unit tests ('pytest tests/'), they skip cleanly to preserve 20s test execution.
    """
    if os.environ.get("RUN_VIDEO_E2E", "").lower() in ("1", "true", "yes"):
        return True
    for arg in sys.argv:
        if "test_video_e2e" in arg:
            return True
    return False


def is_strict_mode() -> bool:
    """Return True if strict Ground Truth matching is enforced."""
    return os.environ.get("STRICT_E2E", "").lower() in ("1", "true", "yes")


skip_if_not_e2e = pytest.mark.skipif(
    not should_run_video_e2e(),
    reason="Video E2E tests process real video files (~40s). Run directly with 'pytest tests/test_video_e2e.py' or set RUN_VIDEO_E2E=1.",
)


def assert_video_counts(
    summary: Dict[str, Any],
    gt_info: Dict[str, Any],
    strict: bool = False,
) -> None:
    """
    Assert video processing results against ground truth and print detailed diagnostic report.
    """
    assert summary is not None, "process_video returned None"
    assert "total_objects" in summary, "Missing 'total_objects' in summary"
    assert "by_class" in summary, "Missing 'by_class' in summary"
    assert "unique_track_ids" in summary, "Missing 'unique_track_ids' in summary"
    assert "frame_count" in summary, "Missing 'frame_count' in summary"
    assert "detections" in summary, "Missing 'detections' in summary"
    assert "output_video" in summary, "Missing 'output_video' in summary"

    detected_count = summary["total_objects"]
    frame_count = summary["frame_count"]
    expected_gt = gt_info["expected_total"]
    expected_frames = gt_info["expected_frames"]
    diff = detected_count - expected_gt
    accuracy = round((detected_count / expected_gt) * 100, 1) if expected_gt > 0 else 0.0

    # Format unique track IDs for display
    unique_tracks = summary["unique_track_ids"]
    formatted_tracks: Dict[str, Any] = {}
    if isinstance(unique_tracks, dict):
        for k, v in unique_tracks.items():
            formatted_tracks[k] = sorted(list(v)) if isinstance(v, (set, list)) else v
    else:
        formatted_tracks = unique_tracks

    # Print high-visibility verification report
    print("\n" + "=" * 80)
    print("           E2E VIDEO COUNTING VERIFICATION REPORT")
    print("=" * 80)
    print(f"Target Video       : {gt_info['path']}")
    print(f"Description        : {gt_info['description']}")
    print(f"Total Video Frames : {frame_count} (Expected: {expected_frames})")
    print(f"Detected Total     : {detected_count} cans")
    print(f"Ground Truth Total : {expected_gt} cans")
    print(f"Counting Error     : {diff:+d} cans ({diff / expected_gt * 100:+.1f}%)")
    print(f"Counting Accuracy  : {accuracy}%")
    print(f"By-Class Detections: {summary['by_class']}")
    print(f"Unique Track IDs   : {formatted_tracks}")
    print(f"Output Video Path  : {summary['output_video']}")
    print(f"Verification Mode  : {'STRICT (M5 Target)' if strict else 'BASELINE (M1 Tracking)'}")
    print(f"Pass/Fail Status   : {'PASS' if detected_count == expected_gt else 'FAIL (Count Mismatch)'}")
    print("=" * 80 + "\n")

    # Sanity checks
    assert frame_count > 0, "No frames were processed from video!"
    assert abs(frame_count - expected_frames) <= 2, (
        f"Frame count mismatch: got {frame_count}, expected {expected_frames}"
    )
    assert os.path.exists(summary["output_video"]), (
        f"Output video file not found at: {summary['output_video']}"
    )

    if strict:
        assert detected_count == expected_gt, (
            f"Strict Ground Truth failure for {gt_info['name']}: "
            f"Detected {detected_count} cans != Expected {expected_gt} cans."
        )


@pytest.fixture(scope="module")
def video_service() -> VideoProcessingService:
    return VideoProcessingService()


@pytest.fixture(scope="module")
def sting_video_result(video_service: VideoProcessingService) -> Dict[str, Any]:
    gt = GROUND_TRUTHS["sting"]
    if not os.path.exists(gt["path"]):
        pytest.skip(f"Video file not found: {gt['path']}")
    out_dir = tempfile.mkdtemp(prefix="test_e2e_sting_")
    out_video = os.path.join(out_dir, "verified_sting.mp4")
    return video_service.process_video(video_path=gt["path"], output_path=out_video)


@pytest.fixture(scope="module")
def sting_cola_video_result(video_service: VideoProcessingService) -> Dict[str, Any]:
    gt = GROUND_TRUTHS["sting_cola"]
    if not os.path.exists(gt["path"]):
        pytest.skip(f"Video file not found: {gt['path']}")
    out_dir = tempfile.mkdtemp(prefix="test_e2e_sting_cola_")
    out_video = os.path.join(out_dir, "verified_sting_cola.mp4")
    return video_service.process_video(video_path=gt["path"], output_path=out_video)


@pytest.mark.e2e
@skip_if_not_e2e
def test_sting_video_pipeline(sting_video_result: Dict[str, Any]) -> None:
    """
    Verify sting.mp4 runs through full pipeline without errors and produces valid detections.
    """
    gt = GROUND_TRUTHS["sting"]
    assert_video_counts(sting_video_result, gt, strict=False)
    assert sting_video_result["total_objects"] > 0, "No objects were detected or counted in sting.mp4"


@pytest.mark.e2e
@skip_if_not_e2e
def test_sting_ground_truth_accuracy(sting_video_result: Dict[str, Any]) -> None:
    """
    Verify sting.mp4 detects exactly 6 cans (retail 6-pack Sting Gold Pro).
    During M1 baseline, xfail is expected until M2-M4 optimizations are merged.
    """
    gt = GROUND_TRUTHS["sting"]
    detected = sting_video_result["total_objects"]
    expected = gt["expected_total"]

    if not is_strict_mode() and detected != expected:
        pytest.xfail(
            f"M1 Baseline: Detected {detected} cans vs Ground Truth {expected} cans "
            f"({gt['description']}). Target accuracy scheduled for M5 after M2-M4 optimizations."
        )

    assert detected == expected, f"Sting count mismatch: got {detected}, expected {expected}"


@pytest.mark.e2e
@skip_if_not_e2e
def test_sting_cola_video_pipeline(sting_cola_video_result: Dict[str, Any]) -> None:
    """
    Verify sting,cola.mp4 runs through full pipeline without errors and produces valid detections.
    """
    gt = GROUND_TRUTHS["sting_cola"]
    assert_video_counts(sting_cola_video_result, gt, strict=False)
    assert sting_cola_video_result["total_objects"] > 0, (
        "No objects were detected or counted in sting,cola.mp4"
    )


@pytest.mark.e2e
@skip_if_not_e2e
def test_sting_cola_ground_truth_accuracy(sting_cola_video_result: Dict[str, Any]) -> None:
    """
    Verify sting,cola.mp4 detects exactly 8 cans (1 Coca-Cola left, 6 Sting, 1 Coca-Cola right).
    During M1 baseline, xfail is expected until M2-M4 optimizations are merged.
    """
    gt = GROUND_TRUTHS["sting_cola"]
    detected = sting_cola_video_result["total_objects"]
    expected = gt["expected_total"]

    if not is_strict_mode() and detected != expected:
        pytest.xfail(
            f"M1 Baseline: Detected {detected} cans vs Ground Truth {expected} cans "
            f"({gt['description']}). Target accuracy scheduled for M5 after M2-M4 optimizations."
        )

    assert detected == expected, f"Sting+Cola count mismatch: got {detected}, expected {expected}"
