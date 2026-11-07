"""
Automated E2E Video Counting Verification Script
================================================
Validates counting accuracy of VideoProcessingService against authoritative Ground Truths:
  - D:\\videotest\\sting.mp4: Ground Truth = 6 cans (shrink-wrapped retail 6-pack Sting Gold Pro, 2x3 grid)
  - D:\\videotest\\sting,cola.mp4: Ground Truth = 8 cans (1 Coca-Cola left, 6 Sting Gold Pro, 1 Coca-Cola right)

Usage:
  python scripts/verify_video_counting.py [--video all|sting|sting_cola] [--strict] [--target-class can]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.video_service import VideoProcessingService

VIDEO_GROUND_TRUTHS: Dict[str, Dict[str, Any]] = {
    "sting": {
        "key": "sting",
        "name": "sting.mp4",
        "path": r"D:\videotest\sting.mp4",
        "ground_truth_count": 6,
        "expected_frames": 177,
        "description": "Shrink-wrapped retail 6-pack of Sting Gold Pro (2x3 grid)",
        "composition": {"Sting Gold Pro": 6},
    },
    "sting_cola": {
        "key": "sting_cola",
        "name": "sting,cola.mp4",
        "path": r"D:\videotest\sting,cola.mp4",
        "ground_truth_count": 8,
        "expected_frames": 256,
        "description": "8 cans in a row (1 Coca-Cola left, 6 Sting center, 1 Coca-Cola right)",
        "composition": {"Sting Gold Pro": 6, "Coca-Cola": 2},
    },
}


def verify_video(
    gt_info: Dict[str, Any],
    service: VideoProcessingService,
    output_dir: str,
    target_class: Optional[str] = None,
    target_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single video and compare results against ground truth."""
    video_path = gt_info["path"]
    vname = gt_info["name"]
    expected_gt = gt_info["ground_truth_count"]
    expected_frames = gt_info["expected_frames"]

    if not os.path.exists(video_path):
        return {
            "key": gt_info.get("key", vname),
            "video_name": vname,
            "video_path": video_path,
            "status": "MISSING_FILE",
            "passed": False,
            "error": f"Video file not found at: {video_path}",
            "ground_truth": expected_gt,
            "detected_count": 0,
            "frame_count": 0,
            "by_class": {},
            "unique_track_ids": [],
            "elapsed_sec": 0.0,
        }

    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, f"verified_{vname}")

    print(f"\n[{vname}] Starting video processing...")
    print(f"  Path: {video_path}")
    print(f"  Ground Truth: {expected_gt} cans ({gt_info['description']})")
    print(f"  Target Filter: class={target_class!r}, name={target_name!r}")

    t0 = time.time()
    try:
        summary = service.process_video(
            video_path=video_path,
            output_path=out_video_path,
            target_class=target_class,
            target_name=target_name,
        )
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        return {
            "key": gt_info.get("key", vname),
            "video_name": vname,
            "video_path": video_path,
            "status": "PROCESSING_ERROR",
            "passed": False,
            "error": str(exc),
            "ground_truth": expected_gt,
            "detected_count": 0,
            "frame_count": 0,
            "by_class": {},
            "unique_track_ids": [],
            "elapsed_sec": round(time.time() - t0, 2),
        }

    detected_count = summary.get("total_objects", 0)
    frame_count = summary.get("frame_count", 0)
    by_class = summary.get("by_class", {})
    unique_tracks = summary.get("unique_track_ids", {})
    # Convert sets inside unique_tracks to sorted lists for JSON serialization
    serialized_tracks: Dict[str, Any] = {}
    if isinstance(unique_tracks, dict):
        for k, v in unique_tracks.items():
            serialized_tracks[k] = sorted(list(v)) if isinstance(v, (set, list)) else v
    elif isinstance(unique_tracks, (set, list)):
        serialized_tracks = sorted(list(unique_tracks))
    else:
        serialized_tracks = unique_tracks

    passed = (detected_count == expected_gt)
    diff = detected_count - expected_gt
    accuracy_pct = round((detected_count / expected_gt) * 100, 1) if expected_gt > 0 else 0.0

    return {
        "key": gt_info.get("key", vname),
        "video_name": vname,
        "video_path": video_path,
        "description": gt_info["description"],
        "expected_frames": expected_frames,
        "frame_count": frame_count,
        "ground_truth": expected_gt,
        "detected_count": detected_count,
        "diff": diff,
        "accuracy_pct": accuracy_pct,
        "passed": passed,
        "status": "PASSED" if passed else "MISMATCH",
        "by_class": by_class,
        "unique_track_ids": serialized_tracks,
        "output_video": out_video_path,
        "elapsed_sec": elapsed,
    }


def print_verification_report(results: List[Dict[str, Any]], strict: bool = False) -> bool:
    """Print a structured verification report and return overall pass/fail boolean."""
    print("\n" + "=" * 90)
    print("                    E2E VIDEO COUNTING VERIFICATION REPORT")
    print("=" * 90)

    header = f"{'Video Name':<16} | {'Frames':<8} | {'GT':<6} | {'Detected':<9} | {'Diff':<6} | {'Accuracy':<9} | {'Status':<8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    all_passed = True
    for res in results:
        vname = res["video_name"]
        frames = f"{res.get('frame_count', 0)}/{res.get('expected_frames', '?')}"
        gt = str(res.get("ground_truth", "?"))
        det = str(res.get("detected_count", 0))
        diff = f"{res.get('diff', 0):+d}" if "diff" in res else "N/A"
        acc = f"{res.get('accuracy_pct', 0.0):.1f}%"
        status = res.get("status", "UNKNOWN")
        passed = res.get("passed", False)
        if not passed:
            all_passed = False

        print(f"{vname:<16} | {frames:<8} | {gt:<6} | {det:<9} | {diff:<6} | {acc:<9} | {status:<8}")

    print("=" * 90)
    print("\nDETAILED BREAKDOWN BY VIDEO:")
    for res in results:
        print(f"\n--- {res['video_name']} ---")
        print(f"  Description       : {res.get('description', 'N/A')}")
        print(f"  Video Path        : {res['video_path']}")
        print(f"  Processed Frames  : {res.get('frame_count')} frames (Elapsed: {res.get('elapsed_sec', 0)}s)")
        print(f"  Ground Truth Count: {res.get('ground_truth')} cans")
        print(f"  Detected Count    : {res.get('detected_count')} cans")
        print(f"  By-Class Counts   : {res.get('by_class')}")
        print(f"  Unique Track IDs  : {res.get('unique_track_ids')}")
        if res.get("error"):
            print(f"  ERROR             : {res['error']}")
        print(f"  Assertion Result  : {'PASS (100% Match)' if res.get('passed') else 'FAIL (Count Mismatch)'}")

    print("\n" + "=" * 90)
    if all_passed:
        print("OVERALL RESULT: [PASS] All video counts match Ground Truth perfectly!")
    else:
        print("OVERALL RESULT: [FAIL] Discrepancies detected between model count and Ground Truth.")
        if not strict:
            print("Note: Running in non-strict mode. Baseline recorded for milestone tracking.")
    print("=" * 90 + "\n")

    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated E2E Video Counting Verification")
    parser.add_argument(
        "--video",
        choices=["all", "sting", "sting_cola"],
        default="all",
        help="Select video to verify (default: all)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict assertions: exit with code 1 if detected count != ground truth",
    )
    parser.add_argument(
        "--target-class",
        default=None,
        help="Target visual class filter (e.g. 'can', default: None)",
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help="Target product name filter (e.g. 'Sting', default: None)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "verification"),
        help="Directory to save output videos",
    )
    parser.add_argument(
        "--json-report",
        default=os.path.join(PROJECT_ROOT, "outputs", "verification_report.json"),
        help="Path to save JSON verification report",
    )

    args = parser.parse_args()

    # Determine targets
    if args.video == "all":
        targets = [VIDEO_GROUND_TRUTHS["sting"], VIDEO_GROUND_TRUTHS["sting_cola"]]
    else:
        targets = [VIDEO_GROUND_TRUTHS[args.video]]

    # Initialize service once
    print("Initializing VideoProcessingService...")
    service = VideoProcessingService()

    results: List[Dict[str, Any]] = []
    for gt_info in targets:
        res = verify_video(
            gt_info=gt_info,
            service=service,
            output_dir=args.output_dir,
            target_class=args.target_class,
            target_name=args.target_name,
        )
        results.append(res)

    all_passed = print_verification_report(results, strict=args.strict)

    # Save JSON report
    if args.json_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_report)), exist_ok=True)
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strict_mode": args.strict,
            "target_class": args.target_class,
            "target_name": args.target_name,
            "all_passed": all_passed,
            "results": results,
        }
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON verification report saved to: {args.json_report}")

    if args.strict and not all_passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
