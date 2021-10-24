import os
import cv2
import Application.utils as utils
import Application.HandTrackingModule as HTM

class_obj = ['Cook', 'How']
raw_path = 'D:\Pictures\Camera Roll\Temp'
preprocessed = os.path.join(raw_path, 'Preprocessed')

detector = HTM.HandDetector()

for obj in class_obj:
    path_class = os.path.join(raw_path, obj)
    path_class_dest = os.path.join(preprocessed, obj)
    i = 0
    for item in os.listdir(path_class):
        img = cv2.imread(os.path.join(path_class, item))
        detected, pts_upper_left, pts_lower_right = detector.find_hands(img)
        if detected:
            file_name = obj + "_" + str(i) + ".jpg"
            img = img[abs(int(pts_lower_right[1])):abs(int(pts_upper_left[1])), abs(int(pts_upper_left[0])):abs(int(pts_lower_right[0]))]
            img = utils.resize_image(img)
            img = utils.skin_segmentation(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(path_class_dest, file_name), img)
            print(file_name)
            i += 1