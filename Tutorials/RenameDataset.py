import cv2
import os

source = 'D:\Pictures\Camera Roll\Temp2'
dest = 'D:\Pictures\Camera Roll\Temp3'

for folder in os.listdir(source):
    folder_source = os.path.join(source, folder)
    folder_dest = os.path.join(dest, folder)
    i = 0
    for item in os.listdir(folder_source):
        img = cv2.imread(os.path.join(folder_source, item))
        if os.path.isdir(folder_dest) is False:
            os.makedirs(folder_dest)
        name = folder + "_" + str(i) + ".jpg"
        cv2.imwrite(os.path.join(folder_dest, name), img)
        i += 1

