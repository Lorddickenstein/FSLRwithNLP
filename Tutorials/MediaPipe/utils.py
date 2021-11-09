import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


""" Returns an image with dimension height by width. """
def resize_image(src_img, height=224, width=224):
    return cv2.resize(src_img, (height, width), interpolation=cv2.INTER_CUBIC)


""" Show image in opencv. """
def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


""" Show image in matplotlib. """
def show_plt_image(src_img):
    plt.imshow(src_img)
    plt.show()


""" Returns the image in binary using Otsu's binarization. """
def get_thresh(src_img):
    return cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


""" Returns an image with morphological transformation using MORPH_CLOSE. """
def morph_image(src_img, method=cv2.MORPH_CLOSE, kernel=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    return cv2.morphologyEx(src_img, method, kernel)


""" Get edges using Canny Edge Detection """
def get_edges(src_img, th1=150, th2=210):
    return cv2.Canny(src_img, th1, th2)


""" Calculate the contours of an image. """
def get_contours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    return cv2.findContours(img.copy(), mode, method)


""" Returns a new image with bounding rectangle. """
def get_bounding_rect(src_img, mask):
    pts = np.column_stack(np.where(mask.transpose() > 0))
    x, y, w, h = cv2.boundingRect(pts)
    return cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 0, 255), 2)


""" Draw contours into an image. """
def draw_contours(src_img, contours):
    return cv2.drawContours(src_img, contours, -1, (255, 255, 255), 2)


""" Draw convex hull from the given hull"""
def draw_convex_hull(hull, src_img):
    pts = np.column_stack(np.where(src_img.transpose() > 0))
    hullpts = cv2.convexHull(pts)
    ((centx, centy), (width, height), angle) = cv2.fitEllipse(hullpts)

    # Draw convex hull on image
    cv2.polylines(hull, [hullpts], True, (0, 0, 255), 1)
    return hull


""" Apply Skin Segmentation"""
def skin_segmentation(src_img):
    # YCrCb pixel upper and lower boundaries
    YCbCr_lower = np.array([0, 135, 80], np.uint8)
    YCbCr_upper = np.array([255, 180, 135], np.uint8)

    # Convert bgr to YCbCr color space
    img_YCbCr = cv2.cvtColor(src_img, cv2.COLOR_BGR2YCrCb)

    # Determine the intensities of YCbCr pixel intensities that fall inside the upper and lower boundaries
    YCrCb_mask = cv2.inRange(img_YCbCr, YCbCr_lower, YCbCr_upper)

    # Apply open morphological transformation
    YCrCb_mask = morph_image(YCrCb_mask, method=cv2.MORPH_OPEN, kernel=(5, 5))

    # Apply close morphological transformation
    YCrCb_mask = morph_image(YCrCb_mask, method=cv2.MORPH_CLOSE, kernel=(9, 9))

    # Apply Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    YCrCb_mask = cv2.dilate(YCrCb_mask, kernel, iterations=2)

    # Blur image to lessen noise
    YCrCb_mask_blur = cv2.medianBlur(YCrCb_mask, 21)

    # Apply mask to the frame
    return cv2.bitwise_and(src_img.copy(), src_img.copy(), mask=YCrCb_mask_blur)


def crop(infile, height, width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j+1) * width, (i + 1) * height)
            yield im.crop(box)


def averagePixels(path):
    r, g, b = 0, 0, 0
    count = 0
    pic = Image.open(path)
    for x in range(pic.size[0]):
        for y in range(pic.size[1]):
            imgData = pic.load()
            tempr, tempg, tempb = imgData[x, y]
            r += tempr
            g += tempg
            b += tempb
            count += 1
    return (r / count), (g / count), (b / count), count


def scale(img, xScale, yScale):
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_CUBIC)
    return res


def convert_frame_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, 1, 1)
        grayframe = scale(gray, 1, 1)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray
