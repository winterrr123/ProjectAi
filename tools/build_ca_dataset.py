"""
Tạo bộ dữ liệu huấn luyện thống nhất cho ca1 (Người) và ca2 (Con Cá)
kết hợp sản phẩm sẵn có để AI nhận diện chuẩn xác khi upload lên web.
"""

import os
import shutil
import cv2
from pathlib import Path
from ultralytics import YOLO

DATASET_DIR = Path("datasets/ca_unified")
if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

for split in ["train", "val"]:
    (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# 1. Trích xuất ca1.mov (Class 1: Người)
print("[1/3] Trích xuất và gán nhãn ca1.mov...")
cap1 = cv2.VideoCapture("uploads/ca1.mov")
total_f1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
step1 = 12  # lấy ~48 frames
coco_model = YOLO("yolov8n.pt")

count_ca1 = 0
for f_idx in range(0, total_f1, step1):
    cap1.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap1.read()
    if not ret:
        break

    res = coco_model.predict(frame, conf=0.35, verbose=False)[0]
    person_boxes = [b for b in res.boxes if int(b.cls[0]) == 0]
    if not person_boxes:
        continue

    split = "val" if (count_ca1 % 5 == 0) else "train"
    img_name = f"ca1_frame_{f_idx:04d}.jpg"
    lbl_name = f"ca1_frame_{f_idx:04d}.txt"

    h, w = frame.shape[:2]
    cv2.imwrite(str(DATASET_DIR / split / "images" / img_name), frame)

    lbl_lines = []
    for b in person_boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        # Class 1: Người
        lbl_lines.append(f"1 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    with open(DATASET_DIR / split / "labels" / lbl_name, "w", encoding="utf-8") as f:
        f.writelines(lbl_lines)

    count_ca1 += 1

cap1.release()
print(f" -> Đã trích xuất {count_ca1} ảnh từ ca1.mov")

# 2. Trích xuất ca2.mp4 (Class 2: Con Cá)
print("[2/3] Trích xuất và gán nhãn ca2.mp4 bằng mô hình Zero-shot...")
cap2 = cv2.VideoCapture("uploads/ca2.mp4")
total_f2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
step2 = 18  # lấy ~52 frames
world_model = YOLO("yolov8s-worldv2.pt")
world_model.set_classes(["fish"])

count_ca2 = 0
for f_idx in range(0, total_f2, step2):
    cap2.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap2.read()
    if not ret:
        break

    res = world_model.predict(frame, conf=0.15, verbose=False)[0]
    boxes = res.boxes
    if len(boxes) == 0:
        continue

    split = "val" if (count_ca2 % 5 == 0) else "train"
    img_name = f"ca2_frame_{f_idx:04d}.jpg"
    lbl_name = f"ca2_frame_{f_idx:04d}.txt"

    h, w = frame.shape[:2]
    cv2.imwrite(str(DATASET_DIR / split / "images" / img_name), frame)

    lbl_lines = []
    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        # Lọc nhiễu quá nhỏ hoặc quá to chiếm hết màn hình
        bw_px = x2 - x1
        bh_px = y2 - y1
        if bw_px < 10 or bh_px < 10 or (bw_px * bh_px) > (w * h * 0.4):
            continue

        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = bw_px / w
        bh = bh_px / h
        # Class 2: Con Cá
        lbl_lines.append(f"2 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    if lbl_lines:
        with open(DATASET_DIR / split / "labels" / lbl_name, "w", encoding="utf-8") as f:
            f.writelines(lbl_lines)
        count_ca2 += 1

cap2.release()
print(f" -> Đã trích xuất {count_ca2} ảnh từ ca2.mp4")

# 3. Thêm một số mẫu sản phẩm từ tokyo_coffee (Class 0: Sản Phẩm)
print("[3/3] Bổ sung mẫu sản phẩm hiện có (tokyo_coffee)...")
coffee_src = Path("datasets/tokyo_coffee")
count_coffee = 0
if coffee_src.exists():
    for split in ["train", "val"]:
        src_imgs = list((coffee_src / split / "images").glob("*.jpg"))
        for p_img in src_imgs[:25]:
            p_lbl = coffee_src / split / "labels" / f"{p_img.stem}.txt"
            if p_lbl.exists():
                shutil.copy(p_img, DATASET_DIR / split / "images" / f"prod_{p_img.name}")
                # Đảm bảo class là 0 (Sản Phẩm)
                with open(p_lbl, "r", encoding="utf-8") as rf:
                    lines = rf.readlines()
                mod_lines = []
                for l in lines:
                    parts = l.strip().split()
                    if parts:
                        parts[0] = "0"
                        mod_lines.append(" ".join(parts) + "\n")
                with open(DATASET_DIR / split / "labels" / f"prod_{p_img.stem}.txt", "w", encoding="utf-8") as wf:
                    wf.writelines(mod_lines)
                count_coffee += 1
print(f" -> Đã sao chép {count_coffee} ảnh sản phẩm cơ sở")

# Ghi file data.yaml
data_yaml_path = DATASET_DIR / "data.yaml"
data_yaml_content = f"""path: {DATASET_DIR.resolve().as_posix()}
train: train/images
val: val/images

nc: 3
names: ['Sản Phẩm', 'Người', 'Con Cá']
"""
with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(data_yaml_content)

print(f"\n[THÀNH CÔNG] Dataset hợp nhất đã tạo tại: {DATASET_DIR}")
print(f"File data.yaml: {data_yaml_path}")
