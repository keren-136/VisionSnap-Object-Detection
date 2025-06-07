import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pyttsx3
import time

print("🚀 Starting GuideMate 🦯...")

# Load SSD MobileNetV2 model
print("🔄 Loading SSD MobileNetV2 model...")
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
print("✅ Object Detection model loaded.")


# Load labels
labels_path = tf.keras.utils.get_file(
    'mscoco_label_map.txt',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
)

# Parse labels
labels = {}
with open(labels_path, 'r') as file:
    for line in file:
        if "id:" in line:
            idx = int(line.split("id:")[1])
        if "display_name:" in line:
            name = line.split("display_name:")[1].strip().replace('"', '')
            labels[idx] = name

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Speech speed

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()
print("✅ Webcam access successful. Press 'q' to quit.")

last_spoken = {}
speak_interval = 2  # seconds
last_time_spoken = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    # Prepare input
    input_tensor = tf.image.convert_image_dtype(frame, tf.uint8)[tf.newaxis, ...]
    detections = model(input_tensor)

    # Process results
    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    h, w, _ = frame.shape
    current_time = time.time()
    spoken_this_frame = set()

    for i in range(len(scores)):
        if scores[i] < 0.5:
            continue

        y1, x1, y2, x2 = boxes[i]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        class_name = labels.get(class_ids[i], "Unknown")

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ({scores[i]*100:.1f}%)",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Voice feedback (every 2 seconds, avoid repeats)
        if current_time - last_time_spoken > speak_interval:
            center_x = (x1 + x2) // 2
            direction = "center"
            if center_x < w // 3:
                direction = "left"
            elif center_x > 2 * w // 3:
                direction = "right"

            object_phrase = f"{class_name} on your {direction}"

            if class_name not in spoken_this_frame:
                if last_spoken.get(class_name) != direction:
                    print(f"🗣️ Saying: {object_phrase}")
                    engine.say(object_phrase)
                    engine.runAndWait()
                    last_spoken[class_name] = direction
                    spoken_this_frame.add(class_name)

    if current_time - last_time_spoken > speak_interval:
        last_time_spoken = current_time

    # Display
    cv2.imshow("GuideMate 🦯 - Press 'q' to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting GuideMate...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🧹 Resources released successfully.")
