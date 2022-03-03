#####################################################################
# Author: Jerson Destacamento, Joshua Cruzat, Rocella Legaspi       #
# Date: October-December                                            #
# Program Title: utils.py                                           #
# Description: Contains all the utility functions needed for image  #
#              processings                                          #
# General System Design: Utility                                    #
# Data structures, Algorithms, Controls: List, Dictionary, File     #
#              Handling, Try Except                                 #
# Requirements: None		                                        #
#####################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(src_img, height=224, width=224, xScale=0, yScale=0):
    """ Returns an image with dimension height by width. """
    return cv2.resize(src_img, (height, width), fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)


def show_image(name, img):
    """ Show image in opencv. """
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_plt_image(src_img):
    """ Show image in matplotlib. """
    plt.imshow(src_img)
    plt.show()


def convert_to_grayscale(src_img):
    """ Convert image into grayscale. """
    return cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)


def get_thresh(src_img):
    """ Returns the image in binary using Otsu's binarization. """
    return cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def morph_image(src_img, method=cv2.MORPH_CLOSE, kernel=(5, 5)):
    """ Returns an image with morphological transformation using MORPH_CLOSE. """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    return cv2.morphologyEx(src_img, method, kernel)


def get_edges(src_img, th1=150, th2=210):
    """ Get edges using Canny Edge Detection """
    return cv2.Canny(src_img, th1, th2)


def get_contours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    """ Calculate the contours of an image. """
    return cv2.findContours(img.copy(), mode, method)


def get_bounding_rect(src_img, mask):
    """ Returns a new image with bounding rectangle. """
    pts = np.column_stack(np.where(mask.transpose() > 0))
    x, y, w, h = cv2.boundingRect(pts)
    return cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 0, 255), 2)


def draw_contours(src_img, contours):
    """ Draw contours into an image. """
    return cv2.drawContours(src_img, contours, -1, (255, 255, 255), 2)


def variance_of_laplacian(image):
    """ Computes the Laplacian and returns the focus measure which is simply the variance of the Laplacian"""
    return cv2.Laplacian(image, cv2.CV_64F).var()


def draw_convex_hull(hull, src_img):
    """ Draw convex hull from the given hull"""
    pts = np.column_stack(np.where(src_img.transpose() > 0))
    hullpts = cv2.convexHull(pts)
    ((centx, centy), (width, height), angle) = cv2.fitEllipse(hullpts)

    # Draw convex hull on image
    cv2.polylines(hull, [hullpts], True, (0, 0, 255), 1)
    return hull


def skin_segmentation(src_img):
    """ Apply Skin Segmentation"""
    # YCrCb pixel upper and lower boundaries
    YCbCr_lower = np.array([0, 135, 85], np.uint8)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    YCrCb_mask = cv2.dilate(YCrCb_mask, kernel, iterations=2)

    # Blur image to lessen noise
    YCrCb_mask_blur = cv2.medianBlur(YCrCb_mask, 21)

    # Apply mask to the frame
    return cv2.bitwise_and(src_img.copy(), src_img.copy(), mask=YCrCb_mask_blur)


def preprocess_image(src_img):
    """ Transform the image into a format that the model expects. """
    skin_mask = skin_segmentation(src_img)
    gray_img = convert_to_grayscale(skin_mask)
    new_size = resize_image(gray_img, height=120, width=120)
    norm_img = new_size.astype('float32')
    norm_img /= 255
    norm_img.reshape(120, 120, 1)
    return new_size, np.expand_dims(norm_img, axis=(-1, 0))


# https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
def detect_blur(src_img):
    """ Returns a boolean if the img is blurry or not """
    focus_measure = variance_of_laplacian(src_img)
    blurriness = True if focus_measure < 500 else False
    return blurriness, focus_measure


def detect_blur2(src_img):
    focus_measure = variance_of_laplacian(src_img)
    return focus_measure


def detect_blur_fft(image, size=60, thresh=10):
    """ Returns a 2-tuple, the magnitude mean and a boolean indicating whether the image is blurry or not"""
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fft_shift[cY - size:cY + size, cX - size:cX + size] = 0
    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)