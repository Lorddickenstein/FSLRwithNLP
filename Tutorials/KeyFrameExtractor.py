# Uses Sobel Operator to extract signs from videos
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('D:\Pictures\Camera Roll\\TestVid1.mp4')
ploty = np.array([])
plotx = np.array([])
i = 0
cnt = 0
while cap.isOpened():
    _, frame = cap.read()
    if not _:
        break

    print(cnt)
    cnt += 1
    currFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
    sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
    currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))
    # currGradient *= 255.0 / currGradient.max()
    # print(sum(currGradient))
    print(currGradient)
    if i != 0:
        diff = sum(currGradient) - sum(prevGradient)
        # print(diff)
        np.append(plotx, i)
        np.append(ploty, diff)

        # if diff == 0:
        #     cv2.imwrite('D:\Documents\Python\images\Signs\\frame_' + str(i), frame)

    prevGradient = currGradient
    cv2.imshow('Sundown', frame)
    i += 1
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
plt.plot(plotx, ploty)
plt.show()
cv2.destroyAllWindows()