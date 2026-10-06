import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from app.services.yolo_service import YOLOService
from app.services.video_service import VideoProcessingService

yolo = YOLOService()

for vname in ['ca1.mov', 'ca2.mp4']:
    cap = cv2.VideoCapture(os.path.join('uploads', vname))
    serv = VideoProcessingService(yolo)
    last_res = None
    for fnum in range(1, 40):
        ret, frame = cap.read()
        if not ret:
            break
        last_res = serv.process_frame(frame, frame_number=fnum)
    cap.release()
    print(f"=== {vname} (sau 40 frames) ===")
    if last_res:
        print(f"  Tổng đối tượng đã xác nhận đếm: {last_res['total']}")
        print(f"  Số đối tượng đang thấy trong khung hình: {len(last_res['detections'])}")
        print(f"  Chi tiết đếm: {last_res['counts']}")
