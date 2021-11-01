# Extract all frames from video
import cv2
import numpy
import os

path = 'D:\Pictures\Camera Roll'
file_name = 'Jers-Late-1'
file_path = os.path.join(path, file_name) + '.mp4'
cap = cv2.VideoCapture(file_path)
i = 0
k = 0
while cap.isOpened():
    _, frame = cap.read()
    if not _:
        break

    if i % 6 == 0:
        name = file_name + '_' + str(k) + '.jpg'
        path = 'D:\Pictures\Camera Roll\Temp'
        path_class = os.path.join(path, file_name)
        if os.path.isdir(path_class) is False:
            os.makedirs(path_class)
        path_image = os.path.join(path_class, name)
        cv2.imwrite(path_image, frame)
        # cv2.imshow('Threshold', frame)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
        k += 1
    i += 1

print(file_name, 'finished extracting frame.')
cap.release()
cv2.destroyAllWindows()