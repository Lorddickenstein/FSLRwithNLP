import os
import cv2

dataset_dir = 'D:\Documents\Thesis\OurDataset\Raw_Dataset'
camera_dir = 'D:\Pictures\Camera Roll\Dataset'
letters = ['A', 'B', 'C', 'D', 'E',
           'F', 'G', 'H', 'I',
           'K', 'L', 'M', 'N', 'O',
           'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

for letter in letters:
    camera_path = os.path.join(camera_dir, letter)
    dataset_path = os.path.join(dataset_dir, "letters\\" + letter)
    i = 0
    for item in os.listdir(camera_path):
        img = cv2.imread(os.path.join(camera_path, item))
        file_name = letter + "_" + str(i) + ".jpg"
        cv2.imwrite(os.path.join(dataset_path, file_name), img)
        i += 1

for number in numbers:
    camera_path = os.path.join(camera_dir, number)
    dataset_path = os.path.join(dataset_dir, "numbers\\" + number)
    i = 0
    for item in os.listdir(camera_path):
        img = cv2.imread(os.path.join(camera_path, item))
        file_name = number + "_" + str(i) + ".jpg"
        cv2.imwrite(os.path.join(dataset_path, file_name), img)
        i += 1
