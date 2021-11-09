# Test an image for Preprocessing
import os
import cv2
import Application.HandTrackingModule as HTM
import matplotlib.pyplot as plt
import numpy as np

def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_plt_image(img):
    plt.imshow(img)
    plt.show()

path = "D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images"
file_name = "L4.jpg"
img = cv2.imread(os.path.join(path, file_name))
show_plt_image(img)

img_copy = img.copy()
detector = HTM.HandDetector()
detected, pts_upper_left, pts_lower_right = detector.find_hands(img)

if detected:
    cv2.rectangle(img_copy, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
    show_plt_image(img_copy)
    ROI = img[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
    show_plt_image(cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB))
    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray, (5, 5), 0)
    norm_img = blur_img.astype('float32')
    norm_img /= 255
    new_size = cv2.resize(norm_img, (224, 224), interpolation=cv2.INTER_CUBIC)
    show_plt_image(new_size)
    new_dim = np.expand_dims(new_size, axis=(0, -1))
