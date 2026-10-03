"""
Công cụ trích xuất ảnh từ video để chuẩn bị dữ liệu gán nhãn (Roboflow / Label Studio).
Sử dụng:
    python tools/extract_frames.py --video sample_test.mp4 --step 15
"""

import argparse
import os
import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2


def extract_frames(video_path: str, output_dir: str = "dataset_raw", step: int = 15):
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video tại '{video_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    count = 0
    saved = 0

    print(f"Bắt đầu trích xuất từ '{video_path}' (Tổng: {total_frames} frames)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame_name = f"{video_name}_frame_{count:05d}.jpg"
            save_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(save_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Hoàn thành! Đã trích xuất {saved} ảnh vào thư mục: '{output_dir}/'")
    print(f"-> Bây giờ bạn có thể kéo cả thư mục '{output_dir}' lên Roboflow.com để gán nhãn!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất ảnh từ video để train AI")
    parser.add_argument("--video", type=str, default="sample_test.mp4", help="Đường dẫn file video")
    parser.add_argument("--output", type=str, default="dataset_raw", help="Thư mục lưu ảnh trích xuất")
    parser.add_argument("--step", type=int, default=15, help="Cứ sau bao nhiêu frame thì lấy 1 ảnh (mặc định 15)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.step)
