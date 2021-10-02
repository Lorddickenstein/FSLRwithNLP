import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)
ret = cap.set(3, 720)
ret = cap.set(4, 480)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def get_thresh(src_img):
    return cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

while cap.isOpened():
    _, frame = cap.read()
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    frame = cv2.cvtColor(frame, 1)
    blank = np.zeros(frame.shape, dtype='uint8')

    """Blur image"""
    blur_img = cv2.GaussianBlur(frame, (35, 35), 0)

    """Thresh"""
    _, th = get_thresh(blur_img)

    """Convert frame to RGB"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    """Get landmarks if exist"""
    results = hands.process(rgb)

    """Draw Landmarks"""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Original', frame)
    cv2.imshow('Landmarks', blank)
    cv2.imshow('Threshold', th)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()