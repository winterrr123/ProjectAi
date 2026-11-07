import cv2
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.video_service import VideoProcessingService

video_path = r'D:\videotest\sting.mp4'
service = VideoProcessingService()
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    frame_number += 1
    
    # Process frame
    res = service.process_frame(
        frame,
        frame_number=frame_number,
        target_class='can',
        target_name='Lon nước tăng lực Sting vàng'
    )
    
    tracked = res['detections']
    new_counts = res.get('new_counts', {})
    
    # Print when new count happens or detections exist
    if new_counts or frame_number % 25 == 0:
        print(f"Frame {frame_number}: {len(tracked)} tracked items, Total so far: {service.counter.get_total_accumulated()}, New: {new_counts}")
        for t in tracked:
            print(f"   tid: {t.get('tracking_id')}, class: {t.get('class_name')}, is_counted: {t.get('is_counted')}, box: ({int(t['x1'])}, {int(t['y1'])}, {int(t['x2'])}, {int(t['y2'])})")

cap.release()
print("\nFinal total accumulated:", service.counter.get_total_accumulated())
print("Global counted track IDs:", service.counter.global_counted_track_ids)
print("ID Aliases:", service.counter.id_aliases)
print("Track hits:", dict(service.counter.track_hits))
