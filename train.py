"""
Script huấn luyện mô hình YOLOv8 nhận diện đa dạng sản phẩm mới.
Sử dụng:
    python train.py --data products.v2i.yolov8/data.yaml --epochs 50 --batch 16
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ultralytics import YOLO


def train_model(data_yaml: str, epochs: int = 50, batch: int = 16, imgsz: int = 416, base_model: str = "yolov8n.pt"):
    if not os.path.exists(data_yaml):
        print(f"Lỗi: Không tìm thấy file cấu hình dữ liệu tại '{data_yaml}'")
        print("Vui lòng xuất dataset từ Roboflow hoặc chỉ định đúng đường dẫn tới file data.yaml!")
        return

    print("=" * 60)
    print(" BẮT ĐẦU HUẤN LUYỆN YOLOv8 NHẬN DIỆN SẢN PHẨM")
    print(f" - File cấu hình dữ liệu: {data_yaml}")
    print(f" - Mô hình nền: {base_model}")
    print(f" - Số vòng học (epochs): {epochs}")
    print(f" - Kích thước ảnh (imgsz): {imgsz}")
    print(f" - Batch size: {batch}")
    print("=" * 60)

    # Khởi tạo mô hình
    model = YOLO(base_model)

    # Chạy train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="train_products",
        project="runs/detect",
        verbose=True,
    )

    # Đường dẫn file best.pt mới
    weights_path = Path("runs/detect/train_products/weights/best.pt")
    if weights_path.exists():
        target_dir = Path("models")
        target_dir.mkdir(exist_ok=True)
        target_model = target_dir / "best.pt"

        # Sao lưu mô hình cũ
        if target_model.exists():
            backup_model = target_dir / "best_backup.pt"
            shutil.copy(target_model, backup_model)
            print(f"-> Đã sao lưu model cũ sang: {backup_model}")

        # Cập nhật model mới
        shutil.copy(weights_path, target_model)
        print("=" * 60)
        print(" HUẤN LUYỆN THÀNH CÔNG!")
        print(f"-> File trọng số tốt nhất: {weights_path}")
        print(f"-> Đã tự động cập nhật vào hệ thống: {target_model}")
        print("-> Khởi động lại hệ thống (python run.py) để áp dụng mô hình mới!")
        print("=" * 60)
    else:
        print("Huấn luyện hoàn thành nhưng không tìm thấy file trọng số xuất ra.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình YOLOv8 nhận diện sản phẩm")
    parser.add_argument("--data", type=str, default="products.v2i.yolov8/data.yaml", help="Đường dẫn file data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện (mặc định 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (mặc định 16)")
    parser.add_argument("--imgsz", type=int, default=416, help="Kích thước ảnh huấn luyện (mặc định 416)")
    parser.add_argument("--base", type=str, default="yolov8n.pt", help="Mô hình nền (yolov8n.pt)")
    args = parser.parse_args()

    train_model(args.data, args.epochs, args.batch, args.imgsz, args.base)
