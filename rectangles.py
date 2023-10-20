import cv2

image = cv2.imread('Images/parking.png')
while True:
    cv2.rectangle(image, (319, 431), (363,533), (255, 100, 100), 2)
    cv2.rectangle(image, (79, 447), (180,491), (255, 100, 100), 2)
    cv2.rectangle(image, (81, 1073), (183, 1039), (255, 100, 100), 2)
    cv2.imshow('Input Image', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap = cv2.VideoCapture('Videos/parking.mp4')

# Pobierz oryginalne wymiary i FPS (klatki na sekundę)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

new_width = int(width * 0.63)
new_height = int(height * 0.63)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Videos/resized_parking4.mp4', fourcc, fps, (new_width, new_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))

    out.write(resized_frame)

cap.release()
out.release()
cv2.destroyAllWindows()





while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))

    out.write(resized_frame)

    cv2.imshow('Resized Video', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
