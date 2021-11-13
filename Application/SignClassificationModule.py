import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
import numpy as np


# Return the sign that corresponds to the index of the predicted class
def find_match(x):
    classes = {'A': 0, 'B': 1, 'Ball': 75, 'Banana': 76, 'Banana-2': 77, 'Banana-3': 78, 'Bread': 79, 'Break': 80,
               'Break-2': 82, 'Bring-2': 83, 'Buy': 84, 'Buy-2': 85, 'Bye': 86, 'C': 2, 'Chair': 52, 'Coconut': 87,
               'Coffee': 89, 'Come': 90, 'Come-2': 91, 'Congratulations': 39, 'Cook': 92, 'D': 3, 'E': 4, 'Egg': 81,
               'Egg-2': 88, 'Egg-3': 97, 'F': 5, 'Fine': 26, 'From': 93, 'From-2': 94, 'G': 6, 'Gabi': 27, 'Get': 95,
               'Get-2': 96, 'Go': 98, 'Go-2': 99, 'Good': 28, 'Great': 40, 'H': 7, 'Hapon': 29, 'Happen': 100,
               'Happen-2': 101, 'He-She': 30, 'Help': 41, 'His-Her': 31, 'How': 102, 'How-2': 103, 'I': 8,
               'I Love You': 32, 'I-Me-My': 33, 'Introduce': 104, 'Introduce-2': 105, 'Invite': 53, 'J': 9, 'K': 10,
               'L': 11, 'Late': 54, 'Late-2': 55, 'Let': 106, 'Let-2': 107, 'Live': 108, 'M': 12, 'Mango': 109,
               'Maybe': 110, 'Meet': 42, 'Mine': 34, 'N': 13, 'Name': 44, 'Nice': 111, 'No': 56, 'No-2': 57, 'Now': 112,
               'O': 14, 'Occupation': 45, 'Office': 113, 'Office-2': 114, 'Our': 58, 'Our-2': 59, 'P': 15, 'Pen': 48,
               'Pray': 46, 'Q': 16, 'R': 17, 'Rest': 47, 'S': 18, 'School': 115, 'Sit': 116, 'Sorry': 60, 'Stand': 49,
               'Store': 117, 'Strawberry': 118, 'Study': 50, 'T': 19, 'Table': 43, 'Tanghali': 35, 'Thank You': 119,
               'Thank You-2': 120, 'That': 61, 'Them': 62, 'This': 63, 'To': 51, 'Today': 121, 'Today-2': 122, 'U': 20,
               'Umaga': 36, 'V': 21, 'W': 22, 'We': 64, 'We-2': 65, 'Welcome': 66, 'Welcome-2': 67, 'What': 123,
               'When': 68, 'Where': 124, 'Which': 125, 'Who': 69, 'Who-2': 70, 'Why': 71, 'Why-2': 72, 'Work': 126,
               'X': 23, 'Y': 24, 'Yes': 73, 'Yesterday': 74, 'You': 37, 'Your': 38, 'Z': 25}
    return list(classes.keys())[list(classes.values()).index(x)]


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
    # score = float("%0.2f" % (prediction[0, class_x] * 100))
    # sign = find_match(class_x)


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


# Load Weights and Compile Model
def load_and_compile(path):
    model = create_model_original()
    model.load_weights(path)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    print(find_match(12))