import tensorflow as tf
from tensorflow import keras
import numpy as np


# Return the sign that corresponds to the index of the predicted class
def find_match(x):
    classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
             5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: ' ',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
             15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
             20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
             25: ' '}
    return classes[x]


# Classify the image using the model
def classify_image(src_img, model):
    predictions = model.predict(src_img)
    top_prediction_indices = np.argsort(predictions)[0, -3:]
    return predictions, top_prediction_indices
    # score = float("%0.2f" % (prediction[0, class_x] * 100))
    # sign = find_match(class_x)

