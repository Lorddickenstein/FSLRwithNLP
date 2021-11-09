import mediapipe as mp
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video_path = 'C:/Users/User/Desktop/videos/DDO'
file_name = 'YES.mp4'

cap = cv2.VideoCapture(os.path.join(video_path, file_name))
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        _, image = cap.read()

        height, width, channel = image.shape
        if not _:
            print("Ignoring empty camera frame.")
            continue;

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.flip(image, 1)
        rgb.flags.writeable = False
        results = hands.process(rgb)

        image.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pinky_mcp, index_mcp = 0, 0

                # for id, lm in enumerate(hand_landmarks.landmark):
                #     h, w, c = image.shape
                #     cx, cy = int(lm.x * w), int(lm.y * h)
                #     if id == 5:
                #         index_mcp = cx
                #     if id == 17:
                #         pinky_mcp = cx
                #
                # if index_mcp > pinky_mcp:
                #     cv2.putText(image, 'Dorsal (back)', (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                #                 (255, 0, 255), 2)
                # if index_mcp < pinky_mcp:
                #     cv2.putText(image, 'Palmar (front)', (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                #                 (255, 0, 255), 2)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()