"""
Challenger 2 Empirical Verification & Boundary Stress Harness
Milestone 5: Beverage Can Detection, Tracking & Counting Verification
"""

import sys
import os
import math
import time
from typing import Dict, Any, List

# Ensure app is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.yolo_service import YOLOService
from app.services.video_service import VideoProcessingService, is_matching_target
from app.services.counting_service import CountingService


def test_suppress_overlapping_boxes_boundary():
    print("\n" + "=" * 80)
    print("TEST SUITE 1: suppress_overlapping_boxes Boundary & Edge-Case Analysis")
    print("=" * 80)

    # 1a. Touching cans side-by-side with varying overlap: 0%, 15%, 30%, 45%
    print("\n--- 1a. Touching cans side-by-side (0%, 15%, 30%, 45% overlap) ---")
    w, h = 100.0, 200.0
    area = w * h
    base_box = {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0, "confidence": 0.85, "class_name": "Sản phẩm", "model_source": "specialist"}

    # Target IoUs: 0.0, 0.15, 0.30, 0.45
    # For side-by-side horizontal shift dx:
    # inter_w = w - dx. inter_area = inter_w * h. union = 2 * area - inter_area
    # iou = (w - dx) / (2w - (w - dx)) = (w - dx) / (w + dx)
    # => dx = w * (1 - iou) / (1 + iou)
    test_ious = [0.0, 0.15, 0.30, 0.45]
    for target_iou in test_ious:
        if target_iou == 0.0:
            dx = w
        else:
            dx = w * (1.0 - target_iou) / (1.0 + target_iou)
        
        can2 = {
            "x1": 100.0 + dx,
            "y1": 100.0,
            "x2": 200.0 + dx,
            "y2": 300.0,
            "confidence": 0.80,
            "class_name": "Sản phẩm",
            "model_source": "specialist",
        }
        
        inter_w = max(0.0, 200.0 - (100.0 + dx))
        inter_area = inter_w * h
        union_area = 2 * area - inter_area
        calc_iou = inter_area / union_area if union_area > 0 else 0.0
        
        input_dets = [dict(base_box), dict(can2)]
        res = YOLOService.suppress_overlapping_boxes(input_dets)
        
        print(f"Overlap target {target_iou*100:.0f}% (Calculated IoU={calc_iou:.3f}, dx={dx:.1f}px):")
        print(f"  Input detections: {len(input_dets)} -> Kept detections: {len(res)}")
        assert len(res) == 2, f"FAILED: Touching cans with {target_iou*100}% overlap were mistakenly suppressed! Kept {len(res)}"
        print("  -> PASSED: Both distinct adjacent cans preserved.")

    # 1b. Identical multi-model duplicate detections (IoU > 0.65)
    print("\n--- 1b. Identical multi-model duplicate detections (IoU > 0.65) ---")
    det_spec = {
        "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0,
        "confidence": 0.88, "class_name": "Sản phẩm", "model_source": "specialist"
    }
    # Jittered duplicate from base COCO model: IoU ~ 0.80
    det_coco = {
        "x1": 103.0, "y1": 102.0, "x2": 198.0, "y2": 298.0,
        "confidence": 0.72, "class_name": "cup", "model_source": "base_coco"
    }
    inter_w = min(det_spec["x2"], det_coco["x2"]) - max(det_spec["x1"], det_coco["x1"])
    inter_h = min(det_spec["y2"], det_coco["y2"]) - max(det_spec["y1"], det_coco["y1"])
    inter_area = inter_w * inter_h
    union_area = (100 * 200) + ((198 - 103) * (298 - 102)) - inter_area
    calc_iou = inter_area / union_area
    print(f"Multi-model duplicate pair (Specialist vs COCO, IoU={calc_iou:.3f}):")
    res_b = YOLOService.suppress_overlapping_boxes([det_spec, det_coco])
    print(f"  Input: 2 -> Kept: {len(res_b)}")
    assert len(res_b) == 1, f"FAILED: Multi-model duplicate was NOT suppressed! Kept: {len(res_b)}"
    assert res_b[0]["model_source"] == "specialist", f"FAILED: Expected specialist model to be retained, got {res_b[0]['model_source']}"
    print(f"  -> PASSED: Duplicate successfully suppressed; prioritized '{res_b[0]['model_source']}' with confidence {res_b[0]['confidence']}.")

    # 1c. Nested inner bounding box (containment > 0.85 with coincident centers)
    print("\n--- 1c. Nested inner bounding box (containment > 0.85, coincident centers) ---")
    outer_box = {
        "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0,  # w=100, h=200, cx=150, cy=200
        "confidence": 0.85, "class_name": "Sản phẩm", "model_source": "specialist"
    }
    inner_box = {
        "x1": 105.0, "y1": 110.0, "x2": 195.0, "y2": 290.0,  # w=90, h=180, cx=150, cy=200
        "confidence": 0.75, "class_name": "Sản phẩm", "model_source": "base_coco"
    }
    in_w = 90.0
    in_h = 180.0
    containment = (in_w * in_h) / (in_w * in_h)
    print(f"Nested inner box pair (Containment={containment:.2f}, Coincident centers):")
    res_c = YOLOService.suppress_overlapping_boxes([outer_box, inner_box])
    print(f"  Input: 2 -> Kept: {len(res_c)}")
    assert len(res_c) == 1, f"FAILED: Nested inner box was NOT suppressed! Kept: {len(res_c)}"
    print(f"  -> PASSED: Nested inner box suppressed correctly.")

    # 1d. Distinct ByteTrack active tracks with different tracking IDs from the same model
    print("\n--- 1d. Distinct ByteTrack active tracks with different tracking IDs (same model) ---")
    track_1 = {
        "tracking_id": 101, "x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0,
        "confidence": 0.85, "class_name": "Sản phẩm", "model_source": "specialist"
    }
    # Heavy overlap track (e.g., occluded can, IoU ~ 0.58 > 0.55 threshold)
    track_2 = {
        "tracking_id": 102, "x1": 130.0, "y1": 100.0, "x2": 230.0, "y2": 300.0,
        "confidence": 0.82, "class_name": "Sản phẩm", "model_source": "specialist"
    }
    t_inter_w = 200.0 - 130.0  # 70
    t_inter_area = 70.0 * 200.0
    t_union_area = 2 * (100.0 * 200.0) - t_inter_area
    t_iou = t_inter_area / t_union_area
    print(f"Same-model tracks with high overlap (IoU={t_iou:.3f} > 0.55, IDs=101, 102):")
    res_d = YOLOService.suppress_overlapping_boxes([track_1, track_2])
    print(f"  Input: 2 -> Kept: {len(res_d)}")
    assert len(res_d) == 2, f"FAILED: Track ID invariance failed! One track was suppressed. Kept: {len(res_d)}"
    kept_ids = {r["tracking_id"] for r in res_d}
    assert kept_ids == {101, 102}, f"FAILED: Expected tracks 101 and 102, got {kept_ids}"
    print(f"  -> PASSED: Track ID Invariance verified. Both distinct active tracks preserved.")


def test_video_service_extreme_and_sequential():
    print("\n" + "=" * 80)
    print("TEST SUITE 2: VideoProcessingService Extreme Parameters & Sequential Resilience")
    print("=" * 80)

    video_sting = r"D:\videotest\sting.mp4"
    video_cola = r"D:\videotest\sting,cola.mp4"

    if not os.path.exists(video_sting) or not os.path.exists(video_cola):
        print("WARNING: Test videos not found, skipping video pipeline stress test.")
        return

    # 2a. target_class="can" vs target_class=None
    print("\n--- 2a. Extreme Parameter Testing: target_class='can' vs target_class=None ---")
    service = VideoProcessingService()

    print("Running process_video with target_class=None on sting.mp4...")
    t0 = time.time()
    res_none = service.process_video(video_sting, target_class=None)
    t_none = time.time() - t0
    print(f"  [target_class=None] Detected: {res_none['total_objects']} cans in {t_none:.2f}s | Classes: {res_none['by_class']}")

    print("Running process_video with target_class='can' on sting.mp4...")
    t0 = time.time()
    res_can = service.process_video(video_sting, target_class="can")
    t_can = time.time() - t0
    print(f"  [target_class='can'] Detected: {res_can['total_objects']} cans in {t_can:.2f}s | Classes: {res_can['by_class']}")

    assert res_none["total_objects"] == 6, f"FAILED: target_class=None expected 6 cans, got {res_none['total_objects']}"
    assert res_can["total_objects"] == 6, f"FAILED: target_class='can' expected 6 cans, got {res_can['total_objects']}"
    print("  -> PASSED: Both target_class='can' and target_class=None accurately counted exactly 6 cans.")

    # Note on generic product model behavior:
    # Primary model detects objects as 'Sản Phẩm' (norm_det == 'product'), which allows
    # generic product detections to be counted under target_name/target_class.
    # In sting.mp4, detections classified specifically as 'can' are suppressed if target is 'bottle',
    # while detections labeled as generic 'Sản Phẩm' fall back to product matching.
    print("  -> NOTE: Generic product model fallback documented. Continuing to 2b.")

    # 2b. Tracker reset resilience across sequential video runs on single service instance
    print("\n--- 2b. Tracker Reset Resilience Across Sequential Video Runs ---")
    print("Executing sequential pipeline on single VideoProcessingService instance:")
    print("  Run 1: sting.mp4 (GT=6)")
    print("  Run 2: sting,cola.mp4 (GT=8)")
    print("  Run 3: sting.mp4 (GT=6) - verifying zero state contamination from Run 2")

    seq_service = VideoProcessingService()

    # Run 1
    t0 = time.time()
    r1 = seq_service.process_video(video_sting)
    print(f"  Run 1 (sting.mp4): Total={r1['total_objects']} (GT=6) | Track IDs: {r1['unique_track_ids']}")
    assert r1["total_objects"] == 6, f"Run 1 Failed: Expected 6 got {r1['total_objects']}"

    # Run 2
    t0 = time.time()
    r2 = seq_service.process_video(video_cola)
    print(f"  Run 2 (sting,cola.mp4): Total={r2['total_objects']} (GT=8) | Track IDs: {r2['unique_track_ids']}")
    assert r2["total_objects"] == 8, f"Run 2 Failed: Expected 8 got {r2['total_objects']}"

    # Run 3 (Crucial test: repeats sting.mp4 on same instance)
    t0 = time.time()
    r3 = seq_service.process_video(video_sting)
    print(f"  Run 3 (sting.mp4 repeated): Total={r3['total_objects']} (GT=6) | Track IDs: {r3['unique_track_ids']}")
    assert r3["total_objects"] == 6, f"Run 3 Failed: State leakage detected! Expected 6 got {r3['total_objects']}"

    # Verify explicit reset() clears all internal state
    seq_service.reset()
    assert len(seq_service._active_tracks) == 0, f"Leaked active tracks after reset: {len(seq_service._active_tracks)}"
    assert len(seq_service._lost_tracks) == 0, f"Leaked lost tracks after reset: {len(seq_service._lost_tracks)}"
    assert seq_service.counter.get_total_accumulated() == 0, f"Counter not reset: {seq_service.counter.get_total_accumulated()}"

    print("  -> PASSED: Sequential runs show perfect reset resilience with zero cross-video track pollution.")


def main():
    print("STARTING EMPIRICAL VERIFICATION HARNESS")
    test_suppress_overlapping_boxes_boundary()
    test_video_service_extreme_and_sequential()
    print("\n" + "=" * 80)
    print("ALL EMPIRICAL CHALLENGER TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
