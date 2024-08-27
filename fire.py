import torch
import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/helloabc/Downloads/best.pt')

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_frame(frame, output_dir, timestamp, severity):
    filename = f"frame_{timestamp}_{severity}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"Saved frame: {filepath}")

def determine_severity(confidence, threshold=0.5):
    return 'High' if confidence >= threshold else 'Low'

def process_video(input_video_path, output_dir, capture_interval):
    # Create directory for saving frames
    create_directory(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {input_video_path}")
        return None, None  # Return None if video couldn't be opened

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * capture_interval)  # Interval in terms of frame count
    frame_count = 0
    last_confidence = None
    last_frame = None

    while True:
        # Read frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get current time for frame capture
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")

        # Convert frame to PIL image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform detection
        results = model(img_pil)

        # Draw results on the frame only for fire
        for _, row in results.pandas().xyxy[0].iterrows():
            if model.names[int(row['class'])] == 'fire':
                x1, y1, x2, y2, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence']
                severity = determine_severity(conf)
                label = f"Fire {conf:.2f} ({severity})"
                color = (0, 0, 255)  # Red color for fire
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Update the last detected confidence and frame
                last_confidence = conf
                last_frame = frame

                # Print the severity
                print(f"Fire detected with confidence {conf:.2f} - Severity: {severity}")

        # Save frame at specified interval
        if frame_count % frame_interval == 0 and last_frame is not None:
            save_frame(last_frame, output_dir, timestamp, severity)

        frame_count += 1

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

    return last_frame, last_confidence  # Return the last frame and confidence
