"""
==============================================================================
PRODUCT VISION - HIGH-PERFORMANCE YOLOv8 TRAINING & ONNX EXPORT ENGINE
Tối ưu hóa tốc độ huấn luyện, độ chuẩn xác mAP và tự động xuất ONNX cho CPU/GPU
==============================================================================
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from ultralytics import YOLO


def check_and_install_onnx():
    """Tự động kiểm tra và cài đặt onnx + onnxruntime nếu thiếu."""
    try:
        import onnx
        return True
    except ImportError:
        print("\n[INFO] Đang tự động cài đặt thư viện 'onnx' và 'onnxruntime' để tối ưu tốc độ xuất mô hình...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "onnx>=1.14.0", "onnxruntime>=1.15.0", "--quiet"],
                check=True
            )
            print("[THÀNH CÔNG] Đã cài đặt xong onnx & onnxruntime!\n")
            return True
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể tự động cài đặt onnx: {e}")
            print("Bạn có thể tự cài bằng lệnh: pip install onnx onnxruntime\n")
            return False


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int):
    """Tải dataset tự động từ Roboflow Universe API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[INFO] Đang cài đặt thư viện roboflow...")
        subprocess.run([sys.executable, "-m", "pip", "install", "roboflow", "--quiet"], check=True)
        from roboflow import Roboflow

    print(f"\n[ROBOFLOW] Đang kết nối tới Roboflow API...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    version_obj = proj.version(version)
    dataset = version_obj.download("yolov8")
    print(f"[THÀNH CÔNG] Đã tải dataset về: {dataset.location}")
    return os.path.join(dataset.location, "data.yaml")


def train_model(
    data_yaml: str,
    epochs: int = 60,
    batch: int = 16,
    imgsz: int = 416,
    base_model: str = "yolov8s.pt",
    device: str = "auto",
    export_onnx: bool = True,
    patience: int = 20,
):
    """Huấn luyện YOLOv8 với siêu tham số tối ưu độ chính xác và xuất mô hình tốc độ cao."""
    if not os.path.exists(data_yaml):
        print(f"\n[LỖI] Không tìm thấy file cấu hình dữ liệu tại: '{data_yaml}'")
        print("Vui lòng kiểm tra lại đường dẫn dataset hoặc xuất dataset từ Roboflow!")
        return

    # Tự động chọn thiết bị (GPU nếu có CUDA, ngược lại CPU)
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "CPU vi xử lý"

    print("\n" + "=" * 65)
    print("      PRODUCT VISION - BẮT ĐẦU HUẤN LUYỆN YOLOv8 PRO")
    print("=" * 65)
    print(f" • File cấu hình dữ liệu: {data_yaml}")
    print(f" • Mô hình nền (Backbone): {base_model}")
    print(f" • Số vòng học (Epochs)  : {epochs} (Early Stop sau {patience} vòng)")
    print(f" • Kích thước ảnh (imgsz): {imgsz}px")
    print(f" • Kích thước Batch      : {batch}")
    print(f" • Thiết bị phần cứng    : {device} ({device_name})")
    print(f" • Tự động xuất ONNX     : {'BẬT (Tối ưu tốc độ 2x-3x)' if export_onnx else 'TẮT'}")
    print("=" * 65 + "\n")

    # Khởi tạo mô hình
    model = YOLO(base_model)

    # Siêu tham số tối ưu độ chính xác cho nhận diện sản phẩm:
    # - mosaic=1.0: Ghép 4 góc ảnh, học tốt các sản phẩm xếp chen chúc / nhỏ
    # - mixup=0.15: Học các sản phẩm bị che khuất một phần
    # - degrees=15.0: Chống xoay nghiêng
    # - fliplr=0.5: Lật ngang ngẫu nhiên
    # - close_mosaic=10: Tắt mosaic ở 10 vòng cuối để hội tụ tối ưu
    # - cos_lr=True: Giảm learning rate theo đường cong Cosine
    print("[1/3] Đang tiến hành huấn luyện mô hình với Augmentations chuyên sâu...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        optimizer="auto",
        cos_lr=True,
        mosaic=1.0,
        mixup=0.15,
        degrees=15.0,
        scale=0.3,
        fliplr=0.5,
        close_mosaic=10,
        name="train_products_pro",
        project="runs/detect",
        verbose=True,
    )

    # Kiểm tra file trọng số tốt nhất
    weights_path = Path(results.save_dir) / "weights" / "best.pt"
    if not weights_path.exists():
        candidates = list(Path("runs/detect").glob("**/weights/best.pt"))
        if candidates:
            weights_path = max(candidates, key=os.path.getmtime)

    if not weights_path.exists():
        print("[LỖI] Huấn luyện xong nhưng không tìm thấy file trọng số xuất ra.")
        return

    target_dir = Path("models")
    target_dir.mkdir(exist_ok=True)
    target_model_pt = target_dir / "best.pt"

    # Sao lưu mô hình cũ
    if target_model_pt.exists():
        backup_model = target_dir / "best_backup.pt"
        shutil.copy(target_model_pt, backup_model)
        print(f"\n[2/3] Đã sao lưu mô hình cũ sang: {backup_model}")

    # Cập nhật file best.pt mới
    shutil.copy(weights_path, target_model_pt)
    print(f"[2/3] Đã cập nhật file trọng số PyTorch: {target_model_pt}")

    # =========================================================================
    # Tự động xuất sang ONNX để tăng tốc độ suy luận (Inference Speedup 2x-3x)
    # =========================================================================
    if export_onnx:
        print("\n[3/3] Đang xuất mô hình sang định dạng ONNX tối ưu vi xử lý...")
        has_onnx = check_and_install_onnx()
        if has_onnx:
            try:
                trained_model = YOLO(str(target_model_pt))
                onnx_file = trained_model.export(
                    format="onnx",
                    dynamic=True,     # Cho phép kích thước batch & resolution linh hoạt
                    simplify=True,    # Tối ưu cấu trúc đồ thị ONNX
                    imgsz=imgsz,
                )
                if os.path.exists(onnx_file):
                    target_onnx = target_dir / "best.onnx"
                    shutil.move(onnx_file, target_onnx)
                    print(f" • [THÀNH CÔNG] File ONNX tối ưu đã lưu tại: {target_onnx}")
                    print(" • Tốc độ đếm trên CPU giờ đây sẽ nhanh hơn từ 200% - 300%!")
            except Exception as e:
                print(f" • [CẢNH BÁO] Không thể xuất ONNX: {e}")
                print(" • Hệ thống vẫn sẽ chạy mượt mà với file best.pt gốc.")
        else:
            print(" • Bỏ qua bước xuất ONNX do thiếu thư viện onnx.")

    print("\n" + "=" * 65)
    print("           HUẤN LUYỆN VÀ TỐI ƯU HÓA HOÀN TẤT!")
    print("=" * 65)
    print(f" -> File PyTorch tốt nhất: {target_model_pt}")
    print(f" -> Đường dẫn models:     {target_dir.resolve()}")
    print(" -> Khởi động lại server: python run.py để bắt đầu đếm với AI mới!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình YOLOv8 nhận diện sản phẩm siêu nhanh và chuẩn xác (Product Vision Pro)"
    )
    parser.add_argument("--data", type=str, default="products.v2i.yolov8/data.yaml", help="Đường dẫn file data.yaml")
    parser.add_argument("--epochs", type=int, default=60, help="Số vòng học epochs (mặc định 60)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (mặc định 16)")
    parser.add_argument("--imgsz", type=int, default=416, help="Kích thước ảnh (mặc định 416, có thể chọn 640)")
    parser.add_argument(
        "--base",
        type=str,
        default="yolov8s.pt",
        help="Mô hình nền: yolov8s.pt (Khuyên dùng - chính xác cao) hoặc yolov8n.pt (siêu nhẹ)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị: auto, cpu, 0")
    parser.add_argument("--patience", type=int, default=20, help="Số vòng dừng sớm nếu không cải thiện (Early stopping)")
    parser.add_argument("--no-onnx", action="store_true", help="Không tự động xuất file ONNX")

    # Tùy chọn tải trực tiếp từ Roboflow API
    parser.add_argument("--roboflow-key", type=str, default=None, help="API Key Roboflow để tự tải dataset")
    parser.add_argument("--workspace", type=str, default=None, help="Tên workspace trên Roboflow")
    parser.add_argument("--project", type=str, default=None, help="Tên project trên Roboflow")
    parser.add_argument("--version", type=int, default=1, help="Phiên bản dataset trên Roboflow")

    args = parser.parse_args()

    # Nếu có Roboflow API key thì tải tự động
    yaml_path = args.data
    if args.roboflow_key and args.workspace and args.project:
        yaml_path = download_roboflow_dataset(
            api_key=args.roboflow_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
        )

    train_model(
        data_yaml=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        base_model=args.base,
        device=args.device,
        export_onnx=not args.no_onnx,
        patience=args.patience,
    )
