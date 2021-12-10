import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
import numpy as np


# Return the sign that corresponds to the index of the predicted class
def find_match(x):
    classes = {0: 'A', 1: 'B', 75: 'Ball', 76: 'Banana', 77: 'Banana', 78: 'Banana', 79: 'Bread', 80: 'Break',
               82: 'Break', 83: 'Bring', 84: 'Buy', 85: 'Buy', 86: 'Bye', 2: 'C', 52: 'Chair', 87: 'Coconut',
               89: 'Coffee', 90: 'Come', 91: 'Come', 39: 'Congratulations', 92: 'Cook', 3: 'D', 4: 'E', 81: 'Egg',
               88: 'Egg', 97: 'Egg', 5: 'F', 26: 'Fine', 93: 'From', 94: 'From', 6: 'G', 27: 'Gabi', 95: 'Get',
               96: 'Get', 98: 'Go', 99: 'Go', 28: 'Good', 40: 'Great', 7: 'H', 29: 'Hapon', 100: 'Happen',
               101: 'Happen', 30: 'He-She', 41: 'Help', 31: 'His-Her', 102: 'How', 103: 'How', 8: 'I',
               32: 'I-Love-You', 33: 'I-Me', 104: 'Introduce', 105: 'Introduce', 53: 'Invite', 9: 'J', 10: 'K',
               11: 'L', 54: 'Late', 55: 'Late', 106: 'Let', 107: 'Let', 108: 'Live', 12: 'M', 109: 'Mango',
               110: 'Maybe', 42: 'Meet', 34: 'Mine', 13: 'N', 44: 'Name', 111: 'Nice', 56: 'No', 57: 'No', 112: 'Now',
               14: 'O', 45: 'Person', 113: 'Office', 114: 'Office', 58: 'Our', 59: 'Our', 15: 'P', 48: 'Pen',
               46: 'Pray', 16: 'Q', 17: 'R', 47: 'Rest', 18: 'S', 115: 'School', 116: 'Sit', 60: 'Sorry', 49: 'Stand',
               117: 'Store', 118: 'Strawberry', 50: 'Study', 19: 'T', 43: 'Table', 35: 'Tanghali', 119: 'Thank-You',
               120: 'Thank-You', 61: 'That', 62: 'Them', 63: 'This', 51: 'To', 121: 'Today', 122: 'Today', 20: 'U',
               36: 'Umaga', 21: 'V', 22: 'W', 64: 'We', 65: 'We', 66: 'Welcome', 67: 'Welcome', 123: 'What',
               68: 'When', 124: 'Where', 125: 'Which', 69: 'Who', 70: 'Who', 71: 'Why', 72: 'Why', 126: 'Work',
               23: 'X', 24: 'Y', 73: 'Yes', 74: 'Yesterday', 37: 'You', 38: 'Your', 25: 'Z'}
    return classes[x]


# Classify the image using the model
def classify_image(src_img, model):
    predictions = model.predict(src_img)
    top_prediction_indices = np.argsort(predictions)[0, -5:]
    top_predictions = []
    for index in top_prediction_indices:
        prediction = find_match(index)
        score = float("%.02f" % (predictions[0, index] * 100))
        top_predictions.append((prediction, score))
    return predictions, top_predictions


# Create the original Sequential Model
def create_model_original():
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


# Create the Sequential Model
def create_model():
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


# Load Weights and Compile Model
def load_and_compile(path):
    model = create_model_original()
    model.load_weights(path)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model


# Load Weights and Compile Model
def load_and_compile_2(path):
    model = create_model()
    model.load_weights(path)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    print(find_match(12))


# def find_match(x):
#     classes = {'A': 0, 'B': 1, 'Ball': 75, 'Banana': 76, 'Banana-2': 77, 'Banana-3': 78, 'Bread': 79, 'Break': 80,
#                'Break-2': 82, 'Bring-2': 83, 'Buy': 84, 'Buy-2': 85, 'Bye': 86, 'C': 2, 'Chair': 52, 'Coconut': 87,
#                'Coffee': 89, 'Come': 90, 'Come-2': 91, 'Congratulations': 39, 'Cook': 92, 'D': 3, 'E': 4, 'Egg': 81,
#                'Egg-2': 88, 'Egg-3': 97, 'F': 5, 'Fine': 26, 'From': 93, 'From-2': 94, 'G': 6, 'Gabi': 27, 'Get': 95,
#                'Get-2': 96, 'Go': 98, 'Go-2': 99, 'Good': 28, 'Great': 40, 'H': 7, 'Hapon': 29, 'Happen': 100,
#                'Happen-2': 101, 'He-She': 30, 'Help': 41, 'His-Her': 31, 'How': 102, 'How-2': 103, 'I': 8,
#                'I Love You': 32, 'I-Me-My': 33, 'Introduce': 104, 'Introduce-2': 105, 'Invite': 53, 'J': 9, 'K': 10,
#                'L': 11, 'Late': 54, 'Late-2': 55, 'Let': 106, 'Let-2': 107, 'Live': 108, 'M': 12, 'Mango': 109,
#                'Maybe': 110, 'Meet': 42, 'Mine': 34, 'N': 13, 'Name': 44, 'Nice': 111, 'No': 56, 'No-2': 57, 'Now': 112,
#                'O': 14, 'Occupation': 45, 'Office': 113, 'Office-2': 114, 'Our': 58, 'Our-2': 59, 'P': 15, 'Pen': 48,
#                'Pray': 46, 'Q': 16, 'R': 17, 'Rest': 47, 'S': 18, 'School': 115, 'Sit': 116, 'Sorry': 60, 'Stand': 49,
#                'Store': 117, 'Strawberry': 118, 'Study': 50, 'T': 19, 'Table': 43, 'Tanghali': 35, 'Thank You': 119,
#                'Thank You-2': 120, 'That': 61, 'Them': 62, 'This': 63, 'To': 51, 'Today': 121, 'Today-2': 122, 'U': 20,
#                'Umaga': 36, 'V': 21, 'W': 22, 'We': 64, 'We-2': 65, 'Welcome': 66, 'Welcome-2': 67, 'What': 123,
#                'When': 68, 'Where': 124, 'Which': 125, 'Who': 69, 'Who-2': 70, 'Why': 71, 'Why-2': 72, 'Work': 126,
#                'X': 23, 'Y': 24, 'Yes': 73, 'Yesterday': 74, 'You': 37, 'Your': 38, 'Z': 25}
#     return list(classes.keys())[list(classes.values()).index(x)]