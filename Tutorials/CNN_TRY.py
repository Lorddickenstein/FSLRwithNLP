from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os  # iterate through the directories
import cv2
import pandas as pd

def test_model_from_dataset(x_train, y_train, x_test, y_test, model_name):
    model = keras.models.load_model('Fingerspelling(16, 32, 64)_(0.5030-0.9015).h5')
    print(y_train[26])
    plt.imshow(x_train[26], cmap='gray')
    plt.show()
    x_train = x_train[26].reshape(-1, 28, 28, 1)
    print(x_train)
    print(x_train.shape)
    print(x_train.ndim)
    prediction = model.predict(x_train)
    print(prediction)
    class_x = np.argmax(prediction, axis=1)
    print(find_match(class_x[0]))

def test_model(img):
    model = keras.models.load_model('FingerSpelling(32, 64, 128)_(0.4652-0.9072).h5')
    prediction = model.predict(img)
    print(prediction)
    class_x = np.argmax(prediction, axis=1)
    print(find_match(class_x[0]))

def find_match(x):
    spell = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: ' ',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: ' ',
    }
    return spell[x]

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def preprocess_image(img):
    """Smoothen img using Gausian blur"""
    # gaussian = cv2.GaussianBlur(img, (11, 11), 0)
    # show_image(gaussian)
    # img = cv2.blur(img, (5, 5), 0)
    # show_image(blur)
    # median = cv2.medianBlur(img, 5)
    # show_image(median)
    """Reshape img to 28x28"""
    img_size = 28
    img = cv2.resize(img, (img_size, img_size))

    """Normalize img"""
    img = img.astype('float32')
    img /= 255
    show_image(img)

    """Expand img into 4d"""
    img = np.expand_dims(img, axis=(0, -1))
    # print(img)
    print("Shape", img.shape)
    print(img.ndim)
    # show_image(img)
    return img

def import_data():
    """@doc Get the train datasets"""
    df_train = pd.read_csv(
        'D:/Documents/Thesis/FSLRwithNLP/Datasets/Fingerspelling/sign_mnist_train/sign_mnist_train.csv')
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
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    """Create the model"""
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(26, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test))
    print(model.evaluate(x_test, y_test))
    return model

def save_model(model, name):
    model.save(name)

path = "D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images"
file_name = "Y2.jpg"
img = cv2.imread(os.path.join(path, file_name), 0)
# show_image(img)
img = preprocess_image(img)
# show_image(img)
test_model(img)

# model_name = "Fingerspelling(16, 32, 64)_(0.5030-0.9015).h5"
# model_name = "FingerSpelling(32, 64, 128)_(0.4652-0.9072).h5"
# x_train, y_train, x_test, y_test = import_data()
# model = create_model(x_train, y_train, x_test, y_test)
# save_model(model, model_name)
# test_model_from_dataset(x_train, y_train, x_test, y_test, model_name)
