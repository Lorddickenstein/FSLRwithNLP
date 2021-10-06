from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import numpy as np
import utils

def import_data():
    DATADIR = "D:\Documents\Thesis\FSLRwithNLP\Datasets\OurDataset\Raw Dataset"
    CATEGORIES = ['A', 'B', 'C', 'D', 'E',
                  'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y',
                  'Z']
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for image in os.listdir(path):
            img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)


def predict_image(src_img, model_name):
    model = keras.models.load_model(model_name)
    prediction = model.predict(src_img)
    class_x = np.argmax(prediction)
    print(class_x)
    print(find_match(class_x))


def find_match(x):
    classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
             5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: ' ',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
             15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
             20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
             25: ' '}
    return classes[x]


import_data()