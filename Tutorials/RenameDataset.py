import cv2
import os

source = 'D:\Pictures\Camera Roll\Temp2'
dest = 'D:\Pictures\Camera Roll\Temp3'
folders = ['Ball', 'Chair', 'Help', 'Sit', 'Strawberry', 'Today-2', 'What', 'When', 'Yes']

for folder in os.listdir(source):
    if folder not in folders:
        continue
    folder_source = os.path.join(source, folder)
    folder_dest = os.path.join(dest, folder)
    i = 0
    for item in os.listdir(folder_source):
        img = cv2.imread(os.path.join(folder_source, item))
        if os.path.isdir(folder_dest) is False:
            os.makedirs(folder_dest)
        name = folder + "_" + str(i) + ".jpg"
        print(name)
        cv2.imwrite(os.path.join(folder_dest, name), img)
        i += 1

