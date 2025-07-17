import cv2
import os

video_path = 'assets/raw_traffic.mp4'
output_folder = 'frames'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps * 2  # 1 frame every 2 seconds

frame_id = 0
saved_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_id}.jpg")
        cv2.imwrite(filename, frame)
        saved_id += 1

    frame_id += 1

cap.release()
