import cv2
import numpy as np
import matplotlib.pyplot as plt


""" Returns an image with dimension height by width. """
def resize_image(src_img, height=224, width=300):
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


""" Get edges using Canny Edge Detection"""
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


""" Apply Skin Segmentation"""
def skin_segmentation(src_img):
    # YCrCb pixel upper and lower boundaries
    YCbCr_lower = np.array([0, 130, 80], np.uint8)
    YCbCr_upper = np.array([255, 180, 140], np.uint8)

    # Convert bgr to YCbCr color space
    img_YCbCr = cv2.cvtColor(src_img, cv2.COLOR_BGR2YCrCb)

    # Determine the intensities of YCbCr pixel intensities that fall inside the upper and lower boundaries
    YCrCb_mask = cv2.inRange(img_YCbCr, YCbCr_lower, YCbCr_upper)

    # Apply open morphological transformation
    YCrCb_mask = morph_image(YCrCb_mask, method=cv2.MORPH_OPEN, kernel=(15, 15))

    # Apply close morphological transformation
    YCrCb_mask = morph_image(YCrCb_mask, method=cv2.MORPH_CLOSE, kernel=(9, 9))

    # Apply Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    YCrCb_mask = cv2.dilate(YCrCb_mask, kernel, iterations=1)

    # Blur image to lessen noise
    YCrCb_mask_blur = cv2.medianBlur(YCrCb_mask, 9)

    # Apply mask to the frame
    return cv2.bitwise_and(src_img.copy(), src_img.copy(), mask=YCrCb_mask_blur)


