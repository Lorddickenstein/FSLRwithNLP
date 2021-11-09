<<<<<<< HEAD
import utils as utils
import HandTrackingModule as HTM
import mediapipe as mp
import cv2
=======
import Application.utils as utils
import Application.HandTrackingModule as HTM
import mediapipe as mp
import cv2
import imutils
>>>>>>> 0593556a0c39c1e11953a098021b60d8238d3ede
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

detector = HTM.HandDetector()

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
<<<<<<< HEAD
=======
    frame = imutils.resize(frame, width=1000)
>>>>>>> 0593556a0c39c1e11953a098021b60d8238d3ede
    height, width, channel = frame.shape
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    print(frame.shape[0], frame.shape[1])

    # Filter lines to make it sharper and smoother
    frame = cv2.bilateralFilter(frame, 5, 50, 100)

    detected, pts_upper_left, pts_lower_right = detector.find_hands(frame.copy(), draw=True)

    if detected:
        # cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
        try:
            roi = frame[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
            roi = utils.skin_segmentation(roi)
<<<<<<< HEAD
            roi = utils.resize_image(roi)
=======
            roi = utils.resize_image(roi, height=120, width=120)
>>>>>>> 0593556a0c39c1e11953a098021b60d8238d3ede
            cv2.imshow('Cropped', roi)
        except Exception:
            pass
    else:
        cv2.putText(frame, "No hands detected...", (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

<<<<<<< HEAD
=======
    cv2.putText(frame, 'Stabilize your arms.', (int(0.65 * width), int(0.05 * height)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

>>>>>>> 0593556a0c39c1e11953a098021b60d8238d3ede
    cv2.imshow('Original', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

cap.release()