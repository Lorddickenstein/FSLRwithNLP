import Application.utils as utils
import Application.HandTrackingModule as HTM
import mediapipe as mp
import cv2
import imutils
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

detector = HTM.HandDetector()

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=1000)
    height, width, channel = frame.shape
    print(height, width)
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    # Filter lines to make it sharper and smoother
    frame = cv2.bilateralFilter(frame, 5, 50, 100)

    detected, pts_upper_left, pts_lower_right = detector.find_hands(frame.copy(), draw=True)

    if detected:
        cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
        print(pts_upper_left, pts_lower_right)
        try:
            # roi = frame[pts_lower_right[1] if pts_lower_right[1] > 0 else 0:pts_upper_left[1] if pts_upper_left[1] > 0 else 0
            #             , pts_upper_left[0] if pts_upper_left[0] > 0 else 0:pts_lower_right[0] if pts_lower_right[0] > 0 else 0]
            roi = frame[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
            roi = utils.skin_segmentation(roi)
            roi = utils.resize_image(roi, height=120, width=120)
            cv2.imshow('Cropped', roi)
        except Exception:
            pass
    else:
        cv2.putText(frame, "No hands detected...", (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.putText(frame, 'Stabilize your arms.', (int(0.65 * width), int(0.05 * height)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow('Original', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

cap.release()