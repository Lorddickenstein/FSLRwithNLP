import cv2
import numpy as np
import Application.utils as utils

cap = cv2.VideoCapture(0)
ret = cap.set(3, 720)
ret = cap.set(4, 480)

while cap.isOpened():
    _, frame = cap.read()
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # cv2.rectangle(frame, )

    blank = np.zeros(frame.shape, dtype='uint8')
    skin_mask = utils.skin_segmentation(frame)

    gray_img = cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, th = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Morph Close
    morph = utils.morph_image(th, kernel=(5, 5))

    # Apply Canny Edge
    edges = utils.get_edges(morph)

    # Get Contours
    contours, hierarchy = utils.get_contours(edges)
    blank = utils.draw_contours(blank, contours)

    # rectangle = utils.get_bounding_rect()
    # x, y, w, h = cv2.boundingRect(contours)
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Normalize img
    norm_img = blank.astype('float32')
    norm_img /= 255

    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    #
    # edges = cv2.Canny(morph, 150, 210)
    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # blank = cv2.drawContours(blank, contours, -1, (255, 255, 255), 2)
    #
    # cv2.imshow('Contour', blank)

    cv2.imshow('Original', frame)
    cv2.imshow('Skin Mask', skin_mask)
    cv2.imshow('Threshold', th)
    cv2.imshow('Contours', blank)
    cv2.imshow('Edges', edges)
    cv2.imshow('Normalized', norm_img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()