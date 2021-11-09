import os
import cv2
<<<<<<< HEAD
import HandTrackingModule as HTM
import utils as utils

path = 'D:\THESIS\Frames1'
dest = 'D:\THESIS\Preprocessed Images1'
file_dir = ['Get-2', 'J', 'Nice',
             'No', 'No-2', 'S', 'That',
             'Umaga', 'We',
             'We-2','Why']
=======
import Application.HandTrackingModule as HTM
import Application.utils as utils

path = 'D:\Pictures\Camera Roll\Temp3'
dest = 'D:\Pictures\Camera Roll\Temp2'
# file_dir = ['Jers-Ago', 'Jers-Allow', 'Jers-Banana', 'Jers-Bread',
#              'Jers-Break', 'Jers-Bring', 'Jers-Buy', 'Jers-Bye',
#              'Jers-Coconut', 'Jers-Coffee', 'Jers-Come', 'Jers-Cook',
#              'Jers-D', 'Jers-Egg-2', 'Jers-From', 'Jers-Get',
#              'Jers-Go', 'Jers-Great', 'Jers-Happen', 'Jers-Happen-2',
#              'Jers-Help', 'Jers-How', 'Jers-How-2', 'Jers-Introduce',
#              'Jers-Invite', 'Jers-Let', 'Jers-Let-2', 'Jers-Maybe',
#              'Jers-Meet', 'Jers-Name', 'Jers-Nice', 'Jers-Nice2',
#              'Jers-Occupation', 'Jers-Pen', 'Jers-Pray', 'Jers-Q',
#              'Jers-Stand', 'Jers-Study', 'Jers-Thank You-2', 'Jers-Today-2',
#              'Jers-Today-2_1', 'Jers-Welcome-2', 'Jers-Where', 'Jers-Which',
#              'Jers-Work']
file_dir = ['test']
>>>>>>> 0593556a0c39c1e11953a098021b60d8238d3ede

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

