import utils as utils
import HandTrackingModule as HTM
import mediapipe as mp
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

detector = HTM.HandDetector()

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
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
            roi = utils.resize_image(roi)
            cv2.imshow('Cropped', roi)
        except Exception:
            pass
    else:
        cv2.putText(frame, "No hands detected...", (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow('Original', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

cap.release()