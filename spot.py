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

cap = cv2.VideoCapture('Videos/resized_parking4.mp4')

def checkparkingspace(preprocessed_frame, original_frame):
    free_counter = 0
    occupied_counter = 0
    for pos in posList:
        x, y, w, h, shape_type = pos

        if shape_type == "rect":
            cropped_frame = preprocessed_frame[y:y+h, x:x+w]
            count = cv2.countNonZero(cropped_frame)
            color = (0, 255, 0) if count < 300 else (0, 0, 255)
            if count < 300:
                free_counter += 1
            else:
                occupied_counter += 1
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), color, 2)

        elif shape_type == "skewed_rect":
            angle = 25
            pts = np.array([
                rotate_point(x, y, x, y - h // 4, angle),
                rotate_point(x, y, x + w, y, angle),
                rotate_point(x, y, x, y + h + h // 4, angle),
                rotate_point(x, y, x - w, y + h, angle)
            ], np.int32)

            mask = np.zeros_like(preprocessed_frame)
            cv2.fillPoly(mask, [pts], 255)
            cropped_frame = cv2.bitwise_and(preprocessed_frame, mask)
            count = cv2.countNonZero(cropped_frame)

            color = (0, 255, 0) if count < 300 else (0, 0, 255)
            if count < 300:
                free_counter += 1
            else:
                occupied_counter += 1
            cv2.polylines(original_frame, [pts], isClosed=True, color=color, thickness=2)

        cv2.putText(original_frame, str(count), (x, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    cv2.putText(original_frame, f'Free: {free_counter}/{len(posList)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(original_frame, f'Occupied: {occupied_counter}/{len(posList)}', (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video or failed to read frame. Exiting.")
        break
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

