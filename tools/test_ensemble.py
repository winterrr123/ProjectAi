import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from app.services.yolo_service import YOLOService

def test_ensemble():
    service = YOLOService()

    videos = {
        "Cam (Oranges)": "uploads/1d1cc40b-5245-4130-91a5-42ce8a7aa794.mp4",
        "Con Cá (Fish)": "uploads/ca2.mp4",
        "Người (Person)": "uploads/ca1.mov",
        "Gói Cà Phê (Coffee)": "uploads/6444194-uhd_3840_2160_24fps.mp4",
    }

    print("\n--- ENSEMBLE VERIFICATION ACROSS VIDEO DATASETS ---")
    for name, vpath in videos.items():
        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"FAILED TO READ: {name}")
            continue

        dets = service.track_infer(frame)
        print(f"\n[+] {name} => {len(dets)} objects detected:")
        for d in dets[:4]:
            print(f"    - {d['class_name']}: conf={d['confidence']}, track_id={d.get('tracking_id')}, source={d.get('model_source')}")

if __name__ == "__main__":
    test_ensemble()
