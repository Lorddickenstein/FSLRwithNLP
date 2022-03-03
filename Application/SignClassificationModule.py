#####################################################################
# Author: Jerson Destacamento, Joshua Cruzat, Rocella Legaspi       #
# Date: October-December                                            #
# Program Title: SignClassificationModule.py                        #
# Description: Provides the Mediapipe hand tracking functionalities #
#              for the main program.                                #
# General System Design: FSL Classification/Prediction, CNN part    #
# Data structures, Algorithms, Controls: List, Dictionary, CNN      #
# Requirements: Models                                              #
#####################################################################

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
import numpy as np


def find_match(x):
    """ Returns the sign that corresponds to the index of the predicted class"""
    classes = {0: 'A', 1: 'B', 75: 'Ball', 76: 'Banana', 77: 'Banana', 78: 'Banana', 79: 'Bread', 80: 'Break',
               82: 'Break', 83: 'Bring', 84: 'Buy', 85: 'Buy', 86: 'Bye', 2: 'C', 52: 'Chair', 87: 'Coconut',
               89: 'Coffee', 90: 'Come', 91: 'Come', 39: 'Congratulations', 92: 'Cook', 3: 'D', 4: 'E', 81: 'Egg',
               88: 'Egg', 97: 'Egg', 5: 'F', 26: 'Fine', 93: 'From', 94: 'From', 6: 'G', 27: 'Evening', 95: 'Get',
               96: 'Get', 98: 'Go', 99: 'Go', 28: 'Good', 40: 'Great', 7: 'H', 29: 'Afternoon', 100: 'Happen',
               101: 'Happen', 30: 'He-She', 41: 'Help', 31: 'His-Her', 102: 'How', 103: 'How', 8: 'I',
               32: 'I-Love-You', 33: 'I-Me', 104: 'Introduce', 105: 'Introduce', 53: 'Invite', 9: 'J', 10: 'K',
               11: 'L', 54: 'Late', 55: 'Late', 106: 'Let', 107: 'Let', 108: 'Live', 12: 'M', 109: 'Mango',
               110: 'Maybe', 42: 'Meet', 34: 'Mine', 13: 'N', 44: 'Name', 111: 'Nice', 56: 'No', 57: 'No', 112: 'Now',
               14: 'O', 45: 'Person', 113: 'Office', 114: 'Office', 58: 'Our', 59: 'Our', 15: 'P', 48: 'Pen',
               46: 'Pray', 16: 'Q', 17: 'R', 47: 'Rest', 18: 'S', 115: 'School', 116: 'Sit', 60: 'Sorry', 49: 'Stand',
               117: 'Store', 118: 'Strawberry', 50: 'Study', 19: 'T', 43: 'Table', 35: 'Noon', 119: 'Thank-You',
               120: 'Thank-You', 61: 'That', 62: 'Them', 63: 'This', 51: 'To', 121: 'Today', 122: 'Today', 20: 'U',
               36: 'Morning', 21: 'V', 22: 'W', 64: 'We', 65: 'We', 66: 'Welcome', 67: 'Welcome', 123: 'What',
               68: 'When', 124: 'Where', 125: 'Which', 69: 'Who', 70: 'Who', 71: 'Why', 72: 'Why', 126: 'Work',
               23: 'X', 24: 'Y', 73: 'Yes', 74: 'Yesterday', 37: 'You', 38: 'Your', 25: 'Z'}
    return classes[x]


def classify_image(src_img, model):
    """ Classifies the image using the model.

        Returns:
            predictions: Numpy Array. An array of predictions.
            top_predictions: List. A list of top 5 predictions.
    """
    predictions = model.predict(src_img)
    top_prediction_indices = np.argsort(predictions)[0, -5:]
    top_predictions = []
    for index in top_prediction_indices:
        prediction = find_match(index)
        score = float("%.02f" % (predictions[0, index] * 100))
        top_predictions.append((prediction, score))
    return predictions, top_predictions


def create_model_1():
    """ Creates the original Sequential Model"""
    model = Sequential()

    # Layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(120, 120, 1), padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(127, activation='softmax'))

    return model


def create_model_2():
    """ Create the Sequential Model"""
    model = Sequential()

    # Layers
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(120, 120, 1), padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(127, activation='softmax'))

    return model


# Create the Sequential Model
def create_model_3():
    model = Sequential()

    # Layers
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(120, 120, 1), padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(127, activation='softmax'))

    return model


def load_and_compile(path, id):
    """ Load Weights and Compile Model"""
    model = create_model_1() if id == 1 else create_model_2() if id == 2 else create_model_3()
    model.load_weights(path)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    print(find_match(12))
