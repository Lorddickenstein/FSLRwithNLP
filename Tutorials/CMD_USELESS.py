import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def show_image(name, img):
    plt.imshow(img, cmap='gray')
    plt.show()
    # cv2.imshow(name, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

path = "D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images"
file_name = "spell.27.jpg"
# file_name = "y2.jpg"
# file_name = "y2.jpg"
img = cv2.imread(os.path.join(path, file_name), 0)
blur_img = cv2.GaussianBlur(img, (11, 11), 0)

# threshold image
_, th = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# th = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
show_image('threshold', th)

# Canny Edge Detection
edges = cv2.Canny(th, 150, 210)
show_image('edges', edges)

""" Find contour"""
# Insert code to get contour of hand

# image gradient
# laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
# show_image('laplacian', laplacian)

# def SaltPepperNoise(edgeImg):
#     count = 0
#     lastMedian = edgeImg
#     median = cv2.medianBlur(edgeImg, 20)
#     while not np.array_equal(lastMedian, median):
#         zeroed = np.invert(np.logical_and(median, edgeImg))
#         edgeImg[zeroed] = 0
#         count += 1
#         if count > 70:
#             break
#         lastMedian = median
#         median = cv2.medianBlur(edgeImg, 3)
#
# edges_ = np.asarray(laplacian, np.uint8)
# SaltPepperNoise(edges_)
# show_image('smoothened', edges_)

# morhp image
# kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
show_image('morph', morph)

result = cv2.bitwise_and(img, img, mask=morph)
show_image('result', result)

img_size = 28
resize_img = cv2.resize(blur_img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
