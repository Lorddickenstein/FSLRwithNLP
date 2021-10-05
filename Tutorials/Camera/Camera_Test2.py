import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)
# ret = cap.set(3, 720)
# ret = cap.set(4, 480)

# HSV pixel upper and lower boundaries
HSV_lower = np.array([0, 15, 0], np.uint8)
HSV_upper = np.array([17, 170, 255], np.uint8)
# HSV_lower = np.array([0, 48, 80], np.uint8)
# HSV_upper = np.array([20, 255, 255], np.uint8)

# HSV pixel upper and lower boundaries
YCbCr_lower = np.array([0, 135, 85], np.uint8)
YCbCr_upper = np.array([255, 180, 135], np.uint8)

while cap.isOpened():
    _, frame = cap.read()
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    # Convert bgr to hsv color space
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Determine the HSV pixel intensities that fall inside the upper and lower boundaries
    HSV_mask = cv2.inRange(img_HSV, HSV_lower, HSV_upper)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN,np.ones((3,3), np.uint8))

    # Convert bgr to YCbCr color space
    img_YCbCr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Determine the intensities of YCbCr pixel intensities that fall inside the upper and lower boundaries
    YCrCb_mask = cv2.inRange(img_YCbCr, YCbCr_lower, YCbCr_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # Merge HSV and YCbCr masks
    merged_mask = cv2.bitwise_and(HSV_mask, YCrCb_mask)
    merged_mask = cv2.medianBlur(merged_mask, 3)
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel)

    # Inverse Mask
    HSV_inverse = cv2.bitwise_not(HSV_mask)
    YCrCb_inverse = cv2.bitwise_not(YCrCb_mask)
    merged_inverse = cv2.bitwise_not(merged_mask)

    # Result Mask
    HSV_result = cv2.bitwise_and(frame.copy(), frame.copy(), mask=HSV_mask)
    YCrCb_result = cv2.bitwise_and(frame.copy(), frame.copy(), mask=YCrCb_mask)
    merged_result = cv2.bitwise_and(frame.copy(), frame.copy(), mask=merged_mask)

    cv2.imshow('HSV Thresh', HSV_inverse)
    cv2.imshow('YCrCb Thresh', YCrCb_inverse)
    cv2.imshow('HSV + YCrCb Thresh', merged_inverse)

    cv2.imshow('HSV', HSV_result)
    cv2.imshow('YCrCb', YCrCb_result)
    cv2.imshow('HSV + YCrCb', merged_result)

    cv2.imshow('Original', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()