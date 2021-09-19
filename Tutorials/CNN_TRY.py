import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os  # iterate through the directories
import cv2
import pandas as pd

"""@doc Get the train datasets"""
df_train = pd.read_csv('D:/Documents/Thesis/FSLRwithNLP/Datasets/Fingerspelling/sign_mnist_train/sign_mnist_train.csv')
# print(df_train.head())

x_train = df_train.drop(columns=['label'])
y_train = df_train[['label']]
# print(x_train.head())
# print(y_train.head())

"""@doc Get the test datasets"""
df_test = pd.read_csv('D:/Documents/Thesis/FSLRwithNLP/Datasets/Fingerspelling/sign_mnist_test/sign_mnist_test.csv')
# print(df_test.head())

x_test = df_test.drop(columns=['label'])
y_test = df_test[['label']]
# print(x_test.head())
# print(y_test.head())

"""Convert to np array"""
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

"""Reshape to 28x28"""
num_rows_train, _ = x_train.shape
num_rows_test, _ = x_test.shape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# print(x_train.shape)

"""Normalize data"""
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

"""Convert to categorical variables"""
y_train = keras.utils.to_categorical(y_train, 26)
y_test = keras.utils.to_categorical(y_test, 26)

"""Create the model"""
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.70))
model.add(keras.layers.Dense(26, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=200, epochs=20, validation_data=(x_test, y_test))
model.save('Fingerspelling_(16, 32, 64).h5')
print(model.evaluate(x_test, y_test))

print(model.predict(x_train[78].reshape(-1, 28, 28), batch_size=1))
# plt.imshow(x_train[0], cmap='gray')
# plt.show()

