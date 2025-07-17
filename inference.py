from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train5/weights/best.pt")

# Run inference on the video
results = model.predict(
    source="assets/traffic_predict.mp4",   # Replace with your actual video path
    conf=0.4,                  # Confidence threshold
    save=True,                 # Save the result video
    save_txt=True,             # Save labels for each frame
    stream=True,               # Show real-time preview
    device=0                   # Use GPU (set to 'cpu' if no GPU)
)
