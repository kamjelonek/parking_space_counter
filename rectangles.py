import cv2

image = cv2.imread('Images/parking.png')
while True:
    cv2.rectangle(image, (319, 431), (363,533), (255, 100, 100), 2)
    cv2.imshow('Input Image', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break