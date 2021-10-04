import cv2
import numpy as np
import mediapipe as mp
import time
import Tutorials.utils as utils

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize mediapipe variables
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing_styles = mp.solutions.drawing_styles

pTime = 0
cTime = 0

def is_overlapping(right_boundary_1, right_boundary_2, left_boundary_1, left_boundary_2):

    # If they are the same square they are not intersecting
    if (right_boundary_1[0] == right_boundary_2[0] or right_boundary_1[1] == right_boundary_2[1] or
    left_boundary_1[0] == left_boundary_2[0] or left_boundary_1[1] == left_boundary_2[1]):
        return False

    # If one rectangle is on the left side of the other
    if (right_boundary_1[0] >= left_boundary_2[0] or left_boundary_1[0] >= right_boundary_2[0]):
        return False

    # If one rectangle is above the other
    if (right_boundary_2[1] >= left_boundary_2[1] or left_boundary_2[1] >= right_boundary_1[1]):
        return False

    # if (right_boundary_1[0] < left_boundary_1[0] and right_boundary_1[1] < left_boundary_1[1]):
    #     return False

    return True

while True:
    _, frame = cap.read()
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    # Filter lines to make it sharper and smoother
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Flip the image
    frame = cv2.flip(frame, 1)

    # Convert frame to rgb for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    rgb.flags.writeable = False
    results = holistic.process(rgb)

    # Get the boundary points on both hands
    height, width, channel = frame.shape
    if results.right_hand_landmarks:
        right_x_pts, right_y_pts = [], []
        for _, lm in enumerate(results.right_hand_landmarks.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            right_x_pts.append(cx)
            right_y_pts.append(cy)
        # Find the max and min points
        right_x_max, right_x_min = max(right_x_pts), min(right_x_pts)
        right_y_max, right_y_min = max(right_y_pts), min(right_y_pts)
        # Get the boundaries
        right_boundary_1 = (right_x_min - 40, right_y_max + 40)
        right_boundary_2 = (right_x_max + 40, right_y_min - 40)
        # cv2.rectangle(frame, right_boundary_1, right_boundary_2, (255, 0, 0), 3)

    if results.left_hand_landmarks:
        left_x_pts, left_y_pts = [], []
        for _, lm in enumerate(results.left_hand_landmarks.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            left_x_pts.append(cx)
            left_y_pts.append(cy)
        # Find the max and min points
        left_x_max, left_x_min = max(left_x_pts), min(left_x_pts)
        left_y_max, left_y_min = max(left_y_pts), min(left_y_pts)
        # Get the boundaries
        left_boundary_1 = (left_x_min - 40, left_y_max + 40)
        left_boundary_2 = (left_x_max + 40, left_y_min - 40)
        # cv2.rectangle(frame, left_boundary_1, left_boundary_2, (255, 0, 0), 3)

    if (results.left_hand_landmarks and results.right_hand_landmarks):
        if is_overlapping(right_boundary_1, right_boundary_2, left_boundary_1, left_boundary_2):
            print('overlapping')
            boundary_1_x = min([left_boundary_1[0], right_boundary_1[0]])
            boundary_1_y = max([left_boundary_1[1], right_boundary_1[1]])
            boundary_2_x = max([left_boundary_2[0], right_boundary_2[0]])
            boundary_2_y = min([left_boundary_2[1], right_boundary_2[1]])
            cv2.rectangle(frame, (boundary_1_x - 20, boundary_1_y + 20), (boundary_2_x + 20, boundary_2_y - 20), (255, 0, 0), 3)
        else:
            cv2.rectangle(frame, left_boundary_1, left_boundary_2, (255, 0, 0), 3)
            cv2.rectangle(frame, right_boundary_1, right_boundary_2, (255, 0, 0), 3)
    else:
        if (results.left_hand_landmarks):
            cv2.rectangle(frame, left_boundary_1, left_boundary_2, (255, 0, 0), 3)
        if (results.right_hand_landmarks):
            cv2.rectangle(frame, right_boundary_1, right_boundary_2, (255, 0, 0), 3)


    # Calculate FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # Show Fps
    cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Original', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()