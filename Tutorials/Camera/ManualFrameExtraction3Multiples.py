# Extract all frames from video
import cv2
import numpy
import os

path = 'D:\Pictures\Camera Roll'
file_name = ['Ago', 'Ago2', 'Ball', 'Ball2', 'Banana', 'Banana2', 'Bread', 'Bread2', 'Coconut', 'Coconut2',
             'Coffee', 'Coffee2', 'Eroplano', 'Eroplano2', 'Late', 'Late2', 'Mango', 'Mango2', 'Office',
             'Office2', 'School', 'School2', 'Store', 'Store2', 'Strawberry', 'Strawberry2', 'That', 'That2',
             'This', 'This2', 'Today', 'Today2', 'We', 'We2', 'Year', 'Year2', 'Yesterday', 'Yesterday2', 'Yesterday3']

for file in file_name:
    file_path = os.path.join(path, file) + '.mp4'
    cap = cv2.VideoCapture(file_path)
    i = 0
    k = 0
    while cap.isOpened():
        _, frame = cap.read()
        if not _:
            break

        if i % 10 == 0:
            name = file + '_' + str(k) + '.jpg'
            path_sign = 'D:\Pictures\Camera Roll\Temp2'
            path_class = os.path.join(path_sign, file)
            if os.path.isdir(path_class) is False:
                os.makedirs(path_class)
            path_image = os.path.join(path_class, name)
            cv2.imwrite(path_image, frame)
            k += 1
        i += 1
    print(file, 'finished extracting frame.')

cap.release()
cv2.destroyAllWindows()