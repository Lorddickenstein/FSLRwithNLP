import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import mediapipe as mp
import time
import Application.utils as utils
import Application.HandTrackingModule as HTM
import matplotlib.pyplot as plt
import warnings
import time
import imutils

tf.get_logger().setLevel('ERROR')
warnings.simplefilter(action='ignore', category=Warning)

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize mediapipe variables
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mp_draw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

detector = HTM.HandDetector()
# model = keras.models.load_model('D:\Documents\Thesis\Experimental_Models\Best so far\Fingerspell_Detector_Experiment5(55-epochs)-accuracy_0.87-val_accuracy_0.84.h5')
# model = keras.models.load_model('D:\Documents\Thesis\Experimental_Models\Fingerspell_Detector_Experiment5(30-epochs).h5')
# model = keras.models.load_model('D:\Documents\Thesis\Experimental_Models\Fingerspell_Detector_Experiment6(10-epochs)-accuracy_0.87-val_accuracy_0.88.h5')
model = keras.models.load_model('D:\Documents\Thesis\Experimental_Models\Y-Why-2_Experiment6(20-epochs)-accuracy_0.87-val_accuracy_0.88.h5')

def find_match2(x):
    classes = {0:'Y', 1:'Why-2'}
    return classes[x]

def find_match(x):
    classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
             5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: ' ',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
             15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
             20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
             25: ' ', 26: 'cook', 27: 'how'}
    return classes[x]

def preprocess_image(src_img):
    skin_mask = utils.skin_segmentation(src_img)
    new_size = cv2.resize(skin_mask, (120, 120), interpolation=cv2.INTER_CUBIC)
    norm_img = new_size.astype('float32')
    norm_img /= 255
    cv2.imshow('what', norm_img)
    return np.expand_dims(norm_img, axis=0)


def classify_image(src_img, model):
    # model = keras.models.load_model('D:\Documents\Thesis\Other Datasets\Model\\Fingerspell_Detector_Experiment1.h5')
    # model = keras.models.load_model('D:\Documents\Thesis\FSLRwithNLP\Tutorials\Models\\Test.h5')
    prediction = model.predict(src_img)
    class_x = np.argmax(prediction)
    print(class_x)
    # score = float("%0.2f" % (max(prediction[0]) * 100))
    score = float("%0.2f" % (prediction[0, class_x] * 100))
    return score, find_match2(class_x)


while True:
    _, frame = cap.read()
    height, width, channel = frame.shape
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    frame = imutils.resize(frame, width=900)
    print(frame.shape[0], frame.shape[1])

    # Filter lines to make it sharper and smoother
    frame = cv2.bilateralFilter(frame, 5, 50, 100)

    detected, pts_upper_left, pts_lower_right = detector.find_hands(frame.copy(), draw=True)

    if detected:
        # cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
        roi = frame[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
        if len(roi) != 0:
            try:
                roi = preprocess_image(roi)
                score, classification = classify_image(roi, model)
                cv2.putText(frame, classification + " " + str(score), (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
            except Exception as exc:
                pass
            # roi = preprocess_image(roi)
            # print(roi.shape)
            # classification = classify_image(roi, model)
            # cv2.putText(frame, classification, (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            # cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
        else:
            cv2.putText(frame, "roi is empty", (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    else:
        cv2.putText(frame, "No hands detected...", (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Calculate FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # Show Fps
    cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Original', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()