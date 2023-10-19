import cv2
import pickle
import numpy as np
import math

# Function to rotate a point
def rotate_point(cx, cy, x, y, angle):
    angle_rad = math.radians(angle)
    x_new = math.cos(angle_rad) * (x - cx) - math.sin(angle_rad) * (y - cy) + cx
    y_new = math.sin(angle_rad) * (x - cx) + math.cos(angle_rad) * (y - cy) + cy
    return int(x_new), int(y_new)

# Try to load shapes, if not exist initialize empty list
try:
    with open('carparkpos', 'rb') as f:
        shapes = pickle.load(f)
except (FileNotFoundError, EOFError):
    shapes = []

current_dims = (44, 102)
current_shape = "rect"

# Function to draw shapes
def draw_shapes(image, shapes):
    for shape in shapes:
        x, y, w, h, shape_type = shape
        if shape_type == "rect":
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 100, 100), 2)
        elif shape_type == "skewed_rect":
            angle = 25
            pts = np.array([
                rotate_point(x, y, x, y - h // 4, angle),
                rotate_point(x, y, x + w, y, angle),
                rotate_point(x, y, x, y + h + h // 4, angle),
                rotate_point(x, y, x - w, y + h, angle)
            ], np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(255, 100, 100), thickness=2)

# Function to handle mouse click events
def mouseClick(events, x, y, flags, params):
    global shapes
    if events == cv2.EVENT_LBUTTONDOWN:
        w, h = current_dims
        shapes.append((x, y, w, h, current_shape))
    elif events == cv2.EVENT_RBUTTONDOWN:
        shapes = [shape for shape in shapes if not (shape[0] <= x <= shape[0] + shape[2] and shape[1] <= y <= shape[1] + shape[3])]

    # Save shapes to file
    with open('carparkpos', 'wb') as f:
        pickle.dump(shapes, f)

# Main loop
while True:
    image = cv2.imread('Images/parking.png')
    draw_shapes(image, shapes)
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', mouseClick)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    elif key & 0xFF == ord('a'):
        current_dims = (44, 102)
        current_shape = "rect"
    elif key & 0xFF == ord('s'):
        current_dims = (102, 44)
        current_shape = "rect"
    elif key & 0xFF == ord('d'):
        current_dims = (44, 92)
        current_shape = "skewed_rect"
