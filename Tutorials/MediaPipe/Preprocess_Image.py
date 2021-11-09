import os
import cv2

import Application.HandTrackingModule as HTM
import Application.utils as utils

path = 'D:\Pictures\Camera Roll\Temp3'
dest = 'D:\Pictures\Camera Roll\Temp2'
file_dir = ['Jers-Ago', 'Jers-Allow', 'Jers-Banana', 'Jers-Bread']

detector = HTM.HandDetector()

# Resize the image into 224x224
def resize_image(src_img, img_size=(224, 224)):
  return cv2.resize(src_img, img_size, interpolation=cv2.INTER_CUBIC)


for file in file_dir:
    path_class = os.path.join(path, file)
    path_class_dest = os.path.join(dest, file)
    if os.path.isdir(path_class_dest) is False:
        os.makedirs(path_class_dest)
    i = 0
    for item in os.listdir(path_class):
        img = cv2.imread(os.path.join(path_class, item))
        detected, pts_upper_left, pts_lower_right = detector.find_hands(img)
        if detected:
            file_name = file + "_" + str(i) + "_new.jpg"
            roi = img[abs(int(pts_lower_right[1])):abs(int(pts_upper_left[1])),
                  abs(int(pts_upper_left[0])):abs(int(pts_lower_right[0]))]
            roi = resize_image(roi)
            roi = utils.skin_segmentation(roi)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(path_class_dest, file_name), roi)
            i += 1
    print(file, 'finished proprocessing.')

