import Application.utils as utils
import Application.HandTrackingModule as HTM
import mediapipe as mp
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video_path = 'D:\Pictures\Camera Roll\Ella'
# file_name = 'we2.mp4'
# file_name = ['Chair', 'Chair2', 'Egg', 'Pen', 'Pen2', 'Table', 'Table2']
file_name = ['AGO']

detector = HTM.HandDetector()

for file in file_name:
    file += '.mp4'
    cap = cv2.VideoCapture(os.path.join(video_path, file))
    while True:
        _, frame = cap.read()
        height, width, channel = frame.shape
        if not _:
            print("Ignoring empty camera frame.")
            continue;

        # print(frame.shape[0], frame.shape[1])
        # print(file)

        # Filter lines to make it sharper and smoother
        frame = cv2.bilateralFilter(frame, 5, 50, 100)

        detected, pts_upper_left, pts_lower_right = detector.find_hands(frame.copy(), draw=True)

        if detected:
            # cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
            try:
                roi = frame[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
                roi = utils.skin_segmentation(roi)
                roi = utils.resize_image(roi, height=80, width=80)
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