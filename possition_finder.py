import cv2
import pickle
width = 44
height = 102
try:
    with open('carparkpos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

posList = []
def mouseClick(events, x,y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x,y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1<x<x1+width and y1<y<y1+height:
                posList.pop(i)
    with open('carparkpos', 'wb') as f:
        pickle.dump(posList, f)
while True:
    image = cv2.imread('Images/parking.png')
    for pos in posList:
        cv2.rectangle(image, (pos[0], pos[1]), (pos[0] + width, pos[1] + height), (255, 100, 100), 2)
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', mouseClick)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
