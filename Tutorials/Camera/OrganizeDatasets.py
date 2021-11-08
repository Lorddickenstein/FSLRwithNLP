# Creates a copy of all dataset and put them in train set folder
# Moves samples to valid and test folders
import random
import shutil
import os
import glob

preprocessed_src = 'D:\Pictures\Camera Roll\Temp2'
train_path = 'D:\Pictures\Camera Roll\Train'
for sign in os.listdir(preprocessed_src):
    path_class_dest = os.path.join(train_path, sign)
    if os.path.isdir(path_class_dest) is False:
        os.makedirs(path_class_dest)

    path_class = os.path.join(preprocessed_src, sign)
    os.chdir(path_class)
    size = len(os.listdir(path_class))
    if size != 0:
        if size > 300:
            size = 300
        for item in random.sample(glob.glob(sign + '_*'), size):
            shutil.copy(item, path_class_dest)