import cv2
import numpy as np
import mediapipe as mp
import imutils
import time
import peakutils
import shutil
import os
import Application.HandTrackingModule as HTM
import Application.utils as utils
import Application.SignClassificationModule as SCM
from tensorflow import keras

# keyframePath = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\keyframes'
# croppedsignsPath = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\keyframes\cropped images'
# model = keras.models.load_model('D:\Documents\Thesis\Experimental_Models\Part2_FSLR_CNN_Model(30-epochs)-accuracy_0.91-val_accuracy_0.91-loss_0.27.h5')
keyframePath = 'E:\\test\\keyframes'
croppedsignsPath = 'E:\\test\\keyframes\\cropped_images'
model = keras.models.load_model('E:\\Part2_FSLR_CNN_Model(30-epochs)-accuracy_0.91-val_accuracy_0.91-loss_0.27.h5')

cap = cv2.VideoCapture(0)
ret = cap.set(3, 720)
ret = cap.set(4, 480)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

detector = HTM.HandDetector()


def scale_img(src_img, xScale, yScale):
    return cv2.resize(src_img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)


def convert_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale_img(gray, xScale=1, yScale=1)
        grayframe = scale_img(gray, xScale=1, yScale=1)
        blur = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, blur


def get_thresh(src_img):
    return cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


isCapturing = False
lstfrm = []
lstdiffMag = []
timeSpans = []
images = []
full_color = []
lastFrame = None
Start_time = 0
text = 'Not Capturing'

while cap.isOpened():
    _, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    height, width, channel = frame.shape
    frameCopy = frame.copy()

    # print(frame.shape[0], frame.shape[1])
    if not _:
        print("Ignoring empty camera frame.")
        continue;

    frame = cv2.cvtColor(frame, 1)
    grayframe, blur_gray = convert_to_grayscale(frame)
    blank = np.zeros(frame.shape, dtype='uint8')

    """Convert frame to RGB"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    """Get landmarks if exist"""
    results = hands.process(rgb)

    """Draw Landmarks"""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if isCapturing:
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time - Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray

    cv2.putText(frameCopy, text, (10, height - int(0.50 * height)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow('Original', frameCopy)
    cv2.imshow('Landmarks', blank)
    key = cv2.waitKey(5)
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        try:
            shutil.rmtree(keyframePath)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        os.makedirs(keyframePath)
        os.makedirs(croppedsignsPath)

        text = 'Capturing'
        Start_time = time.process_time()
        lastFrame = blur_gray
        isCapturing = True
    elif key == ord('e'):
        if isCapturing:
            y = np.array(lstdiffMag)
            base = peakutils.baseline(y, 2)
            indices = peakutils.indexes(y - base, min_dist=3)
            cnt = 1
            sentence = []
            prevWord = ''
            for index in indices:
                extracted_frame = full_color[index]
                detected, pts_upper_left, pts_lower_right = detector.find_hands(extracted_frame)

                if detected:
                    roi = extracted_frame[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
                    try:
                        text = 'not detected'
                        img_crop, roi = utils.preprocess_image(roi)

                        """ Blur detecion using FFT"""
                        # resized = imutils.resize(img_crop, width=500)
                        # (mean, is_blurry) = utils.detect_blur_fft(resized, size=60, thresh=-35)
                        # text = "Blurry ({:.2f})" if is_blurry else "Not Blurry ({:.2f})"
                        # text = text.format(mean)
                        # predictions, top_predictions = SCM.classify_image(roi, model)
                        # print(top_predictions[2:])
                        # score = max(top_predictions, key=lambda x: x[1])
                        # word = top_predictions[4][0]
                        # if word != prevWord:
                        #     cv2.imwrite(os.path.join(croppedsignsPath, str(cnt) + "_" + word + '_' + text + '.jpg'), img_crop)
                        #     sentence.append(word)
                        # prevWord = word
                        resized = imutils.resize(img_crop, width=500)
                        (mean, is_blurry) = utils.detect_blur_fft(resized, size=60, thresh=-35)
                        text = "Blurry ({:.2f})" if is_blurry else "Not Blurry ({:.2f})"
                        text = text.format(mean)
                        if not is_blurry:
                            predictions, top_predictions = SCM.classify_image(roi, model)
                            print(top_predictions[2:])
                            score = max(top_predictions, key=lambda x: x[1])
                            word = top_predictions[4][0]
                            if word != prevWord:
                                cv2.imwrite(os.path.join(croppedsignsPath, str(cnt) + "_" + word + '_' + text + '.jpg'),
                                            img_crop)
                                sentence.append(word)
                            prevWord = word
                    except Exception as exc:
                        pass
                    cv2.imwrite(os.path.join(keyframePath, 'keyframe_' + text + str(cnt) + '.jpg'), extracted_frame)
                    cnt += 1
            print(sentence)
            lstfrm = []
            lstdiffMag = []
            timeSpans = []
            images = []
            full_color = []
            isCapturing = False
            text = 'Not Capturing'

cap.release()
cv2.destroyAllWindows()
