import cv2
import pickle
import numpy as np
import math

def rotate_point(cx, cy, x, y, angle):
    angle_rad = math.radians(angle)
    x_new = math.cos(angle_rad) * (x - cx) - math.sin(angle_rad) * (y - cy) + cx
    y_new = math.sin(angle_rad) * (x - cx) + math.cos(angle_rad) * (y - cy) + cy
    return int(x_new), int(y_new)

try:
    with open('carparkpos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

cap = cv2.VideoCapture('Videos/resized_parking3.mp4')

width, height = 44, 102

def checkparkingspace(preprocessed_frame, original_frame):
    counter = 0
    for pos in posList:
        x, y, w, h, shape_type = pos  # Unpack all values, including shape type

        if shape_type == "rect":
            cropped_frame = preprocessed_frame[y:y+h, x:x+w]
            count = cv2.countNonZero(cropped_frame)
            color = (0, 255, 0) if count < 900 else (0, 0, 255)
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), color, 2)

        elif shape_type == "skewed_rect":
            angle = 25  # The angle you used when drawing
            pts = np.array([
                rotate_point(x, y, x, y - h // 4, angle),
                rotate_point(x, y, x + w, y, angle),
                rotate_point(x, y, x, y + h + h // 4, angle),
                rotate_point(x, y, x - w, y + h, angle)
            ], np.int32)
            cv2.polylines(original_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # Add your logic here to check if the parking space is free or not

        cv2.putText(original_frame, str(count), (x, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    cv2.putText(original_frame, f'Free: {counter}/{len(posList)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue


    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_scale, (3, 3), 1)
    frame_threshold = cv2.adaptiveThreshold(blur, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 25, 16)
    median_blur = cv2.medianBlur(frame_threshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    frame_dilate = cv2.dilate(median_blur, kernel, iterations=1)

    checkparkingspace(frame_dilate, frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import pickle
import numpy as np

# Load saved positions
try:
    with open('carparkpos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

# Start capturing video
cap = cv2.VideoCapture('Videos/resized_parking3q.mp4')

# Rectangle dimensions
width, height = 44, 102

# Main loop to read each video frame and apply the parking space check
while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Draw rectangles directly on the original frame
    for pos in posList:
        x, y, _, _, _ = pos
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
