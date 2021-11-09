# Creates a copy of all dataset and put them in train set folder
# Moves samples to valid and test folders
import random
import shutil
import os
import glob

preprocessed_src = 'D:\Pictures\Camera Roll\Temp2'
train_path = 'D:\Pictures\Camera Roll\Temp\Train'
valid_path = 'D:\Pictures\Camera Roll\Temp\Valid'
test_path = 'D:\Pictures\Camera Roll\Temp\Test'
file_name = ['Come-2', 'D', 'Happen', 'Happen-2', 'He-She', 'Introduce-2', 'Them', 'Welcome-2', 'Work']

def copy_to_train_folder():
    for sign in os.listdir(preprocessed_src):
        if sign not in file_name:
            continue
        path_class_dest = os.path.join(train_path, sign)
        if os.path.isdir(path_class_dest) is False:
            os.makedirs(path_class_dest)

        print(sign)

        path_class = os.path.join(preprocessed_src, sign)
        os.chdir(path_class)
        size = len(os.listdir(path_class))
        if size != 0:
            if size > 300:
                size = 300
            for item in random.sample(glob.glob(sign + '_*'), size):
                shutil.copy(item, path_class_dest)


def count_items_in_train():
    for sign in os.listdir(train_path):
        path_class = os.path.join(train_path, sign)
        print(sign, len(os.listdir(path_class)))


def take_samples():
    valid_size = 50
    test_size = 20

    for sign in os.listdir(train_path):
        if sign not in file_name:
            continue
        path_class = os.path.join(train_path, sign)
        os.chdir(path_class)
        print(sign)

        valid_path_dest = os.path.join(valid_path, sign)
        if os.path.isdir(valid_path_dest) is False:
            os.makedirs(valid_path_dest)

        test_path_dest = os.path.join(test_path, sign)
        if os.path.isdir(test_path_dest) is False:
            os.makedirs(test_path_dest)

        if len(os.listdir(path_class)) != 0:
            for item in random.sample(glob.glob(sign + '_*'), valid_size):
                shutil.move(item, valid_path_dest)

            for item in random.sample(glob.glob(sign + '_*'), test_size):
                shutil.move(item, test_path_dest)


def verify_totals():
    for sign in os.listdir(train_path):
        train_path_experiment = os.path.join(train_path, sign)
        valid_path_experiment = os.path.join(valid_path, sign)
        test_path_experiment = os.path.join(test_path, sign)
        print(sign, "\t\t\t\t", len(os.listdir(train_path_experiment)), len(os.listdir(valid_path_experiment)),
              len(os.listdir(test_path_experiment)))


# copy_to_train_folder()
# count_items_in_train()
take_samples()
verify_totals()
