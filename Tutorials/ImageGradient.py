import cv2
import numpy as np
from matplotlib import pyplot as plt

def resize_image(src_img, height=224, width=224, xScale=0, yScale=0):
    return cv2.resize(src_img, (height, width), fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)

def show_img(name, src_img):
    cv2.imshow(name, src_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images\A_68.jpg', 0)
img = resize_image(img)
# show_img('Resized', img)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))
print(currGradient)
