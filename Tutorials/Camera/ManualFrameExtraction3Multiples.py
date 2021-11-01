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

path = 'D:\Pictures\Camera Roll\Ella'
path_sign_dest = 'D:\Pictures\Camera Roll\Temp3'
file_name = ['AGO', 'ALLOW', 'BANANA', 'BANANA-2', 'BREAD',
             'BREAK', 'BREAK-2', 'BRING', 'BRING-2', 'BUY',
             'BUY-2', 'BYE', 'CHAIR1', 'CHAIR2', 'COCONUT1',
             'COCONUT2', 'COFFEE', 'COME', 'COME-2', 'COOK',
             'E', 'EGG-2', 'FROM', 'FROM-2', 'GET',
             'GO', 'GO-2', 'GREAT', 'HAPPEN', 'HAPPEN-2',
             'HELP', 'HOW', 'HOW-2', 'INTRODUCE', 'INTRODUCE-2',
             'LET', 'MAYBE', 'MEET', 'NAME', 'NICE',
             'OCCUPATION', 'PEN', 'PRAY', 'STAND', 'STUDY1',
             'STUDY2', 'THANK YOU-2', 'TODAY', 'TODAY-2', 'WELCOME-2',
             'WHERE', 'WHICH', 'WORK']
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

            if i % 7 == 0:
                path_class = os.path.join(path_sign_dest, file)
                if os.path.isdir(path_class) is False:
                    os.makedirs(path_class)
                detected, pts_upper_left, pts_lower_right = detector.find_hands(frame)
                if detected:
                    name = 'Ella-' + file + "_" + str(k) + "_new.jpg"
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