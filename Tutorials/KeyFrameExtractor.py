# Uses Sobel Operator to extract signs from videos
import cv2
import numpy as np
import matplotlib.pyplot as plt
import Application.HandTrackingModule as HTM
import Application.utils as utils

detector = HTM.HandDetector()

cap = cv2.VideoCapture('D:\Pictures\Camera Roll\\Test6.mp4')

frm_num = []
frm_gradients = []
prevGradient = np.array([])
i = 0
while cap.isOpened():
    _, frame = cap.read()
    if not _:
        break

    currFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
    sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
    currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

    if i % 5 == 0 and i != 0:

        frm_diff = cv2.absdiff(currGradient, prevGradient)
        frm_sum = cv2.sumElems(frm_diff)
        print(frm_sum, i)

        frm_gradients.append(frm_sum)
        frm_num.append(i)

        cv2.imshow('Sobelx', sobelx)
        cv2.imshow('Sobely', sobely)
        cv2.imshow('Gradient', currGradient)
        cv2.imshow('Sundown', currFrame)

    prevGradient = currGradient
    i += 1
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
plt.plot(frm_num, frm_gradients)
plt.show()
cv2.destroyAllWindows()