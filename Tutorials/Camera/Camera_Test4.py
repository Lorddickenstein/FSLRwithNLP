import cv2
import numpy as np
import time
import imutils
import Application.utils as utils

# Open the camera
cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

is_capturing = False
text = 'Not Capturing'
full_color = []
while True:
    _, frame = cap.read()
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    # Filter lines to make it sharper and smoother
    frame = cv2.bilateralFilter(frame, 5, 50, 100)

    frame = imutils.resize(frame, width=1000)
    height, width, channel = frame.shape
    print(height, width)
    frameCopy = frame.copy()


    if is_capturing:
        gray = utils.convert_to_grayscale(frameCopy)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
        currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))






    key = cv2.waitKey(3) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        text = 'Capturing Frames'
    elif key == ord('e'):
        text = 'Not Capturing'

    cv2.putText(frame, text, (10, int(0.95 * height)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Calculate FPS and show Fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Original', frame)

cap.release()
cv2.destroyAllWindows()