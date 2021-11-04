import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
import os
import time
import imutils
import numpy as np
import mediapipe as mp
import time
import Application.utils as utils
import Application.HandTrackingModule as HTM
import Application.SignClassificationModule as SCM
import matplotlib.pyplot as plt
import warnings

tf.get_logger().setLevel('ERROR')
warnings.simplefilter(action='ignore', category=Warning)

def start_application():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Time variables
    pTime = 0
    cTime = 0
    prevTime = time.time()
    currTime = 0
    ctrTime = 0

    # Model
    model_path = 'D:\Documents\Thesis\Experimental_Models\Best so far'
    model_name = 'Fingerspell_Detector_Experiment5(55-epochs)-accuracy_0.87-val_accuracy_0.84.h5'
    model = load_model(os.path.join(model_path, model_name))

    # Detection variables
    sequence = []
    sentence = []
    threshold = 0.5


    while cap.isOpened():
        _, frame = cap.read()
        height, width, channel = frame.shape
        if not _:
            print("Ignoring empty camera frame.")
            continue;
        frame = imutils.resize(frame, width=1000)

        # Filter lines to make it sharper and smoother
        frame = cv2.bilateralFilter(frame, 5, 50, 100)








        # Calculate time lapse in seconds
        currTime = time.time()
        if currTime - prevTime >= 2.0:
            ctrTime += 1
            prevTime = currTime

        cv2.putText(frame, str(ctrTime), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # Show Fps
        # cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Camera', frame)
        k = cv2.waitKey(5)
        if k == ord('q') or k == 27:
            break


if __name__ == "__main__":
    start_application()