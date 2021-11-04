import cv2
import numpy as np
import mediapipe as mp
import time
import Application.utils as utils

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize mediapipe variables
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mp_draw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    _, frame = cap.read()
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    print(frame_number)

    # Filter lines to make it sharper and smoother
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Flip the image
    frame = cv2.flip(frame, 1)

    # Convert frame to rgb for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLMs in results.multi_hand_landmarks:
            x_pts = []
            y_pts = []
            for id, lm in enumerate(handLMs.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                x_pts.append(cx)
                y_pts.append(cy)

            # Find the max and min points
            y_max, y_min, x_max, x_min = max(y_pts), min(y_pts), max(x_pts), min(x_pts)
            cv2.rectangle(frame, (x_min - 20, y_max + 20), (x_max + 20, y_min - 20), (255, 0, 0), 3)

            mp_draw.draw_landmarks(frame, handLMs, mpHands.HAND_CONNECTIONS)
    else:
        # INSERT CODE TO DISPLAY IF NO HAND IS DETECTED
        pass

    # Calculate FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # Show Fps
    cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Original', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()