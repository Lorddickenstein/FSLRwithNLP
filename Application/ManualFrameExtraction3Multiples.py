# Extract all frames from video
import cv2
import numpy
import os

path = 'D:\THESIS\Videos1'
file_name = ['Get-2', 'J', 'Nice',
             'No', 'No-2', 'S', 'That',
             'Umaga', 'We',
             'We-2','Why']

print(len(file_name))
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

            if i % 10 == 0:
                name = file + '_' + str(k) + '.jpg'
                path_sign = 'D:\THESIS\Frames1'
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