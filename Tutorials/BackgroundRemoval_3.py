import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
ret = cap.set(3, 720)
ret = cap.set(4, 480)

bg_cap_flag = False

def remove_background(frame):
    fgMask = bgSub.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    res = cv2.bitwise_and(frame, frame, mask=fgMask)
    return res

while cap.isOpened():
    _, frame = cap.read()
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    if bg_cap_flag:
        object = remove_background(frame)
        cv2.imshow('Subject', object)

    frame = cv2.cvtColor(frame, 1)
    cv2.imshow('Original', frame)
    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('b'):
        bgSub = cv2.createBackgroundSubtractorMOG2(0, 50)
        # bgSub = cv2.createBackgroundSubtractorKNN(0, 50)
        bg_cap_flag = True
        print('Background is captured')
        bgMask = bgSub.apply(frame)

cv2.destroyAllWindows()