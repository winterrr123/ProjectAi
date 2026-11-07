import cv2
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.video_service import VideoProcessingService

video_path = r'D:\videotest\sting,cola.mp4'
service = VideoProcessingService()
cap = cv2.VideoCapture(video_path)

frame_number = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    frame_number += 1
    
    res = service.process_frame(
        frame,
        frame_number=frame_number,
        target_class='can',
        target_name='Lon nước tăng lực Sting vàng'
    )
    
    tracked = res['detections']
    new_counts = res.get('new_counts', {})
    
    if new_counts or frame_number % 50 == 0:
        print(f"Frame {frame_number}: {len(tracked)} tracked items, Total: {service.counter.get_total_accumulated()}, New: {new_counts}")
        for t in tracked:
            print(f"   tid: {t.get('tracking_id')}, is_counted: {t.get('is_counted')}, is_new: {t.get('is_new')}, box: ({int(t['x1'])}, {int(t['y1'])}, {int(t['x2'])}, {int(t['y2'])})")

cap.release()
print("\nFinal total accumulated:", service.counter.get_total_accumulated())
print("Global counted track IDs:", service.counter.global_counted_track_ids)
print("ID Aliases:", service.counter.id_aliases)
