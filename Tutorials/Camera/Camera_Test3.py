import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret = cap.set(3, 720)
ret = cap.set(4, 480)

# HSV pixel upper and lower boundaries
HSV_lower = np.array([0, 15, 0], np.uint8)
HSV_upper = np.array([17, 170, 255], np.uint8)

# YCrCb pixel upper and lower boundaries
YCbCr_lower = np.array([0, 135, 85], np.uint8)
YCbCr_upper = np.array([255, 180, 135], np.uint8)

while cap.isOpened():
    _, frame = cap.read()
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    blank = np.zeros(frame.shape, np.uint8)

    # Convert bgr to YCbCr color space
    img_YCbCr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Determine the intensities of YCbCr pixel intensities that fall inside the upper and lower boundaries
    YCrCb_mask = cv2.inRange(img_YCbCr, YCbCr_lower, YCbCr_upper)

    # Apply open morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    YCrCb_mask_morphed = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, kernel)

    # Apply close morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    YCrCb_mask_morphed = cv2.morphologyEx(YCrCb_mask_morphed, cv2.MORPH_CLOSE, kernel)

    # Apply Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    YCrCb_mask_morphed = cv2.dilate(YCrCb_mask_morphed, kernel, iterations=1)

    # Blur image to lessen noise
    YCrCb_mask_blur = cv2.medianBlur(YCrCb_mask_morphed, 9)

    # Apply mask to the frame
    YCrCb_result = cv2.bitwise_and(frame.copy(), frame.copy(), mask=YCrCb_mask_blur)

    # Convert to Grayscale
    gray_img = cv2.cvtColor(YCrCb_result, cv2.COLOR_BGR2GRAY)

    # Blur result
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Threshold
    _, th = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(th, 150, 210)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank = cv2.drawContours(blank, contours, -1, (255, 255, 255), 2)

    # cv2.imshow('YCbCr', YCrCb_mask)
    # cv2.imshow('YCbCr Blur', YCrCb_mask_blur)
    cv2.imshow('Original', frame)
    cv2.imshow('YCbCr Final', YCrCb_result)
    cv2.imshow('Thresh', th)
    cv2.imshow('Contours', blank)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

