import cv2
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.yolo_service import YOLOService

for vname in [r'D:\videotest\sting.mp4', r'D:\videotest\sting,cola.mp4']:
    cap = cv2.VideoCapture(vname)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n=== Video: {vname} | {w}x{h} | {total_frames} frames ({total_frames/fps:.1f}s) ===")
    
    yolo = YOLOService()
    
    # Check max detections in any single frame across the video
    max_dets_frame = 0
    max_dets_list = []
    
    for f_idx in range(0, total_frames, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok: break
        
        # Test raw inference with lower threshold (e.g. 0.15)
        dets = yolo.track_infer(frame, confidence_threshold=0.15)
        # Filter non-noise
        valid_dets = [d for d in dets if d.get('class_name') not in ('Người', 'Ghế', 'dining table', 'chair', 'person', 'Con Cá')]
        
        if len(valid_dets) > max_dets_frame:
            max_dets_frame = len(valid_dets)
            max_dets_list = [(d['class_name'], d['raw_name'], d['confidence'], (int(d['x1']), int(d['y1']), int(d['x2']), int(d['y2']))) for d in valid_dets]
            
    print(f"Max products detected in a single frame simultaneously: {max_dets_frame}")
    for item in max_dets_list:
        print(f"   {item}")
    cap.release()
