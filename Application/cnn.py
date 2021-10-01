from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
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

import_data()