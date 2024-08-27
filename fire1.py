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
    # Ensure output_dir is a valid directory path
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a valid string representing a directory path")

    # Generate a filename based on timestamp and severity
    filename = f"frame_{timestamp}_{severity}.jpg"
    filepath = os.path.join(output_dir, filename)

    # Save the image to the specified filepath
    cv2.imwrite(filepath, frame)
    print(f"Saved frame: {filepath}")

def determine_severity(confidence, threshold=0.5):
    return 'High' if confidence >= threshold else 'Low'

def process_image(image, is_fire_detected):
    # Create directory for saving frames
    output_dir = "/Users/helloabc/Desktop/frames"  # Replace with your actual path
    create_directory(output_dir)

    # Convert image to PIL image for model processing
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Perform detection
    results = model(img_pil)

    # Initialize variables
    fire_detected = False
    last_confidence = None

    # Process detection results
    for _, row in results.pandas().xyxy[0].iterrows():
        if model.names[int(row['class'])] == 'fire':
            fire_detected = True
            x1, y1, x2, y2, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence']
            severity = determine_severity(conf)
            label = f"Fire {conf:.2f} ({severity})"
            color = (0, 0, 255)  # Red color for fire
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the frame if fire is detected
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_frame(image, output_dir, timestamp, severity)

            # Update the last detected confidence
            last_confidence = conf

            # Print the severity
            print(f"Fire detected with confidence {conf:.2f} - Severity: {severity}")

    # Return the confidence and severity
    return last_confidence, determine_severity(last_confidence) if last_confidence is not None else 'None'
