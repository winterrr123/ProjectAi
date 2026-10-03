import cv2
import os
import shutil
import numpy as np
from pathlib import Path

# Setup dataset directories
dataset_dir = Path("datasets/tokyo_coffee")
if dataset_dir.exists():
    shutil.rmtree(dataset_dir)

for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
    (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

# Bounding box trajectories (normalized coordinates in 640x360 space)
# We define anchor keyframes (frame_idx -> list of [x1, y1, x2, y2] in 640x360)
# Bag 1: enters before 0, leaves around frame 630
bag1_keyframes = {
    0:   [250, 30, 410, 320],
    100: [225, 33, 385, 325],
    200: [195, 37, 355, 335],
    300: [165, 40, 335, 340],
    400: [120, 43, 290, 345],
    500: [60,  48, 220, 350],
    600: [0,   60, 130, 355],
    640: [0,   80, 50,  355],
}

# Bag 2: visible from frame 0 in background to frame 1690 exiting foreground
bag2_keyframes = {
    0:    [390, 0,  465, 85],
    150:  [390, 0,  475, 95],
    300:  [390, 0,  485, 120],
    450:  [390, 0,  495, 150],
    600:  [390, 0,  500, 190],
    750:  [380, 0,  495, 220],
    900:  [365, 0,  490, 245],
    1050: [330, 10, 470, 275],
    1200: [270, 20, 420, 310],
    1350: [200, 30, 360, 330],
    1450: [155, 35, 320, 340],
    1550: [100, 45, 250, 350],
    1650: [10,  60, 150, 355],
    1695: [0,   75, 80,  355],
}

# Bag 3: enters background around frame 720, visible through frame 1699
bag3_keyframes = {
    720:  [390, 0, 435, 25],
    850:  [390, 0, 450, 50],
    1000: [390, 0, 465, 80],
    1150: [390, 0, 480, 110],
    1300: [390, 0, 490, 140],
    1450: [385, 0, 498, 180],
    1550: [380, 0, 495, 205],
    1650: [370, 0, 492, 235],
    1699: [365, 0, 490, 245],
}

# Bag 4: enters background around frame 1580
bag4_keyframes = {
    1580: [390, 0, 435, 25],
    1650: [390, 0, 445, 45],
    1699: [390, 0, 455, 65],
}

def interpolate_box(keyframes, frame):
    sorted_frames = sorted(keyframes.keys())
    if frame < sorted_frames[0] or frame > sorted_frames[-1]:
        return None
    if frame in keyframes:
        return keyframes[frame]
    
    # Find surrounding keys
    for i in range(len(sorted_frames) - 1):
        f1 = sorted_frames[i]
        f2 = sorted_frames[i + 1]
        if f1 <= frame <= f2:
            t = (frame - f1) / (f2 - f1)
            b1 = keyframes[f1]
            b2 = keyframes[f2]
            return [
                b1[0] + t * (b2[0] - b1[0]),
                b1[1] + t * (b2[1] - b1[1]),
                b1[2] + t * (b2[2] - b1[2]),
                b1[3] + t * (b2[3] - b1[3]),
            ]
    return None

cap = cv2.VideoCapture("uploads/6444194-uhd_3840_2160_24fps.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

sample_frames = list(range(0, 1690, 40))
print(f"Sampling {len(sample_frames)} frames from video...")

W, H = 640, 360

for idx, fnum in enumerate(sample_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_resized = cv2.resize(frame, (W, H))
    
    boxes = []
    for kf in [bag1_keyframes, bag2_keyframes, bag3_keyframes, bag4_keyframes]:
        b = interpolate_box(kf, fnum)
        if b is not None:
            # Check minimum size
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            if bw > 15 and bh > 15:
                boxes.append(b)
    
    # Split into train (80%) and val (20%)
    split = "val" if (idx % 5 == 0) else "train"
    
    img_name = f"coffee_frame_{fnum:04d}.jpg"
    lbl_name = f"coffee_frame_{fnum:04d}.txt"
    
    cv2.imwrite(str(dataset_dir / split / "images" / img_name), frame_resized)
    
    # Write YOLO format: <class> <x_center> <y_center> <width> <height>
    lbl_lines = []
    for b in boxes:
        x1, y1, x2, y2 = b
        x1 = max(0.0, min(float(W), x1))
        x2 = max(0.0, min(float(W), x2))
        y1 = max(0.0, min(float(H), y1))
        y2 = max(0.0, min(float(H), y2))
        
        xc = ((x1 + x2) / 2.0) / W
        yc = ((y1 + y2) / 2.0) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        lbl_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        
    with open(dataset_dir / split / "labels" / lbl_name, "w") as f:
        f.writelines(lbl_lines)

# Write data.yaml
data_yaml_content = f"""path: {dataset_dir.resolve().as_posix()}
train: train/images
val: val/images

nc: 1
names: ['Gói Cà Phê']
"""
with open(dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
    f.write(data_yaml_content)

print("Dataset generated successfully at:", dataset_dir)
cap.release()
