# Extract all frames from video
import cv2
import numpy
import os
import Application.HandTrackingModule as HTM
import Application.utils as utils

detector = HTM.HandDetector()

# Resize the image into 224x224
def resize_image(src_img, img_size=(224, 224)):
  return cv2.resize(src_img, img_size, interpolation=cv2.INTER_CUBIC)

path = 'D:\Pictures\Camera Roll'
path_sign_dest = 'D:\Pictures\Camera Roll\Temp'
file_name = ['Late']
print(len(file_name,))
for file in file_name:
    file_path = os.path.join(path, file) + '.mp4'
    if os.path.exists(file_path) is False:
        print(file_path, 'not exists.')
    else:
        cap = cv2.VideoCapture(file_path)
        i = 0
        k = 0
        while cap.isOpened():
            _, frame = cap.read()
            if not _:
                break

            if i % 8 == 0:
                path_class = os.path.join(path_sign_dest, file)
                if os.path.isdir(path_class) is False:
                    os.makedirs(path_class)
                detected, pts_upper_left, pts_lower_right = detector.find_hands(frame)
                if detected:
                    name = 'Jers_new' + file + "_" + str(k) + ".jpg"
                    path_image = os.path.join(path_class, name)
                    roi = frame[abs(int(pts_lower_right[1])):abs(int(pts_upper_left[1])),
                          abs(int(pts_upper_left[0])):abs(int(pts_lower_right[0]))]
                    roi = resize_image(roi)
                    roi = utils.skin_segmentation(roi)
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(path_image, roi)
                    k += 1
            i += 1
        print(file, 'finished extracting frame.')
        cap.release()

cv2.destroyAllWindows()