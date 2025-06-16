from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO('yolov8n.pt') 

# Initialize Deep SORT
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture('Cars.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)[0]
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)

        # Only track cars, trucks, buses (COCO class IDs)
        if class_id in [2, 5, 7]:  
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracking results
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Car Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
