from ultralytics import YOLO
import cv2
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
import os
import numpy as np
from typing import Dict, Set, Tuple

def calculate_iou(box1, box2):
    """Calculate intersection over union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection)

class VehicleTracker:
    def __init__(self, line_orientation, line_pos, line_tol=10):
        self.tracked_vehicles: Dict[int, Dict] = {}
        self.next_id = 0
        self.iou_threshold = 0.2
        self.counted_vehicles: Set[int] = set()
        self.class_counts = defaultdict(int)
        self.total_count = 0
        self.line_orientation = line_orientation
        self.line_pos = line_pos
        self.line_tol = line_tol  # Tolerance in pixels for counting

    def update(self, detections, frame_count: int):
        current_boxes = []
        current_classes = []
        for det in detections:
            current_boxes.append(det.xyxy[0].cpu().numpy())
            current_classes.append(int(det.cls[0]))
        # Match detections to tracks
        matched_detections = set()
        for track_id, track_info in self.tracked_vehicles.items():
            last_box = track_info['boxes'][-1]
            best_iou = self.iou_threshold
            best_detection_idx = -1
            for i, current_box in enumerate(current_boxes):
                if i in matched_detections:
                    continue
                iou = calculate_iou(last_box, current_box)
                if iou > best_iou:
                    best_iou = iou
                    best_detection_idx = i
            if best_detection_idx >= 0:
                self.tracked_vehicles[track_id]['boxes'].append(current_boxes[best_detection_idx])
                self.tracked_vehicles[track_id]['last_seen'] = frame_count
                matched_detections.add(best_detection_idx)
        # Add new tracks for unmatched detections
        for i in range(len(current_boxes)):
            if i not in matched_detections:
                self.tracked_vehicles[self.next_id] = {
                    'boxes': [current_boxes[i]],
                    'class_id': current_classes[i],
                    'last_seen': frame_count
                }
                self.next_id += 1
        self._count_vehicles()
        self._remove_old_tracks(frame_count)

    def _count_vehicles(self):
        for track_id, track_info in self.tracked_vehicles.items():
            if track_id in self.counted_vehicles:
                continue
            boxes = track_info['boxes']
            class_id = track_info['class_id']
            # Check for crossing the line between consecutive frames
            for i in range(1, len(boxes)):
                prev_box = boxes[i-1]
                curr_box = boxes[i]
                prev_center_x = (prev_box[0] + prev_box[2]) / 2
                prev_center_y = (prev_box[1] + prev_box[3]) / 2
                curr_center_x = (curr_box[0] + curr_box[2]) / 2
                curr_center_y = (curr_box[1] + curr_box[3]) / 2
                if self.line_orientation == "horizontal":
                    # Check if center crosses the line
                    if ((prev_center_y < self.line_pos and curr_center_y >= self.line_pos) or
                        (prev_center_y >= self.line_pos and curr_center_y < self.line_pos)):
                        self.counted_vehicles.add(track_id)
                        self.class_counts[class_id] += 1
                        self.total_count += 1
                        break
                else:
                    if ((prev_center_x < self.line_pos and curr_center_x >= self.line_pos) or
                        (prev_center_x >= self.line_pos and curr_center_x < self.line_pos)):
                        self.counted_vehicles.add(track_id)
                        self.class_counts[class_id] += 1
                        self.total_count += 1
                        break

    def _remove_old_tracks(self, current_frame: int, max_age: int = 30):
        ids_to_remove = [track_id for track_id, track_info in self.tracked_vehicles.items()
                         if current_frame - track_info['last_seen'] > max_age]
        for track_id in ids_to_remove:
            del self.tracked_vehicles[track_id]

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")  # Replace with your actual model path

# Get video properties for output
source_path = "assets/raw_traffic.mp4"  # Replace with your actual video path
cap = cv2.VideoCapture(source_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --- User input for line orientation and position ---
print("Choose counting line orientation:")
print("1. Horizontal (top to bottom)")
print("2. Vertical (left to right)")
orientation_choice = input("Enter 1 for horizontal or 2 for vertical: ").strip()

if orientation_choice == "1":
    line_orientation = "horizontal"
else:
    line_orientation = "vertical"

position_percent = input(f"Enter line position as percentage (0-100, where 0 is {'top' if line_orientation=='horizontal' else 'left'}, 100 is {'bottom' if line_orientation=='horizontal' else 'right'}, 50 is center): ").strip()
try:
    position_percent = int(position_percent)
    if not (0 <= position_percent <= 100):
        position_percent = 50
except:
    position_percent = 50

# --- Initialize tracker ---
if line_orientation == "horizontal":
    line_pos = int(height * position_percent / 100)
else:
    line_pos = int(width * position_percent / 100)

tracker = VehicleTracker(line_orientation, line_pos)

# Get class names from the model
class_names = model.names

# Run inference on the video
source_path = "assets/raw_traffic.mp4"  # Replace with your actual video path
cap = cv2.VideoCapture(source_path)

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer for output
output_path = 'assets/traffic_predict_with_counts.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Run inference on the frame
    results = model.predict(
        source=frame,
        conf=0.4,      # Confidence threshold
        device=0       # Use GPU (set to 'cpu' if no GPU)
    )[0]

    # Update tracker with new detections
    tracker.update(results.boxes, frame_count)

    # Process detections and draw boxes
    boxes = results.boxes
    for box in boxes:
        # Get class ID and confidence
        class_id = int(box.cls[0])
        confidence = box.conf[0]

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_names[class_id]} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Draw the chosen line on each frame ---
    if line_orientation == "horizontal":
        cv2.line(frame, (0, line_pos), (width, line_pos), (255, 0, 0), 2)
    else:
        cv2.line(frame, (line_pos, 0), (line_pos, height), (0, 0, 255), 2)
    
    # Add count information to frame
    y_pos = 30
    cv2.putText(frame, f"Total Unique Vehicles: {tracker.total_count}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    y_pos += 40
    
    # Display per-class counts only
    for class_id, count in tracker.class_counts.items():
        class_name = class_names[class_id]
        cv2.putText(frame, f"{class_name}: {count}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_pos += 30

    # Write frame to output video
    out.write(frame)

    # Display frame (optional)
    cv2.imshow('Vehicle Detection and Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Create results directory if it doesn't exist
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Get current timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Convert class IDs to names for final results
final_class_counts = {
    class_names[class_id]: count 
    for class_id, count in tracker.class_counts.items()
}

# Prepare data for saving
results_data = {
    "total_vehicles": tracker.total_count,
    "class_counts": final_class_counts,
    "timestamp": datetime.now().isoformat(),
    "video_source": source_path
}

# Save as JSON
json_path = os.path.join(results_dir, f"vehicle_counts_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(results_data, f, indent=4)

# Save as Excel
excel_path = os.path.join(results_dir, f"vehicle_counts_{timestamp}.xlsx")
df = pd.DataFrame([
    {"Vehicle Type": class_name, "Count": count}
    for class_name, count in final_class_counts.items()
])
df.loc[len(df)] = ["Total", tracker.total_count]  # Add total row

# Create Excel writer with formatting
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Vehicle Counts', index=False)
    
    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Vehicle Counts']
    
    # Add formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D3D3D3',
        'border': 1
    })
    total_format = workbook.add_format({
        'bold': True,
        'bg_color': '#E8E8E8',
        'border': 1
    })
    
    # Apply formats
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    
    # Format the total row
    last_row = len(df)
    worksheet.set_row(last_row, None, total_format)
    
    # Adjust column widths
    worksheet.set_column('A:A', 20)  # Vehicle Type column
    worksheet.set_column('B:B', 15)  # Count column

# Print final counts
print("\nFinal Vehicle Counts:")
print(f"Total Unique Vehicles: {tracker.total_count}")
print("\nPer-class counts:")
for class_id, count in tracker.class_counts.items():
    print(f"{class_names[class_id]}: {count}")

print(f"\nResults saved to:")
print(f"JSON: {json_path}")
print(f"Excel: {excel_path}")
