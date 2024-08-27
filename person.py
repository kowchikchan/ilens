import cv2
import numpy as np
import os
import time

# Load YOLO
net = cv2.dnn.readNet(
    "/home/logicfocus/ilens/docs/yolov3.weights",
    "/home/logicfocus/ilens/docs/yolov3.cfg"
)
# net = cv2.dnn.readNet("/Users/helloabc/ilensLatest1/src/main/resources/scripts/peopleCount/Testing/yolov3.weights", "/Users/helloabc/ilensLatest1/src/main/resources/scripts/peopleCount/Testing/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("/home/logicfocus/ilens/docs/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video or use webcam
#video = "/Users/helloabc/Downloads/B.mp4"
rtsp_url = 'rtsp://192.168.0.191:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
cap = cv2.VideoCapture(rtsp_url)  # Replace "your_video.mp4" with 0 for webcam

# Create a directory to save captured frames
# saving_dir = "/Users/helloabc/Desktop/person captures"
saving_dir = "/home/logicfocus/ilens"

os.makedirs(saving_dir, exist_ok=True)

# Set the interval (in seconds) to capture frames
capture_interval = 1
last_capture_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        last_capture_time = current_time

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6 and class_id == 0:  # Only keep high confidence and person class
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

        count = 0
        person_detected = False
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                count += 1
                person_detected = True

        if person_detected:
            # Save the frame with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            frame_path = os.path.join(saving_dir, f"person_detected_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Person detected at {timestamp}! Frame saved at: {frame_path}")

cap.release()
cv2.destroyAllWindows()
