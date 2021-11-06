import cv2
import numpy as np
import matplotlib.pyplot as plt
import Application.utils as utils
import Application.HandTrackingModule as HTM
import Application.SignClassificationModule as SCM
from tensorflow import keras
import imutils
import shutil
import os

# Open the camera
cap = cv2.VideoCapture(0)
detector = HTM.HandDetector()

text = 'Not Capturing'
is_capturing = False

GRADIENT_THRESH_VALUE = 3.1
TEN_MILLION = 10000000.0
THRESHOLD = 40.0

keyframes_arr, crop_frm_arr, frm_arr, frm_num_arr, frm_gradients = [], [], [], [], []
prevGradient = np.array([])
prev_frm_sum, start_index, end_index, frm_num = TEN_MILLION, 0, 0, 0

figures_path = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\Figures'
keyframes_path = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\keyframes'
cropped_img_path = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\keyframes\cropped'
model = keras.models.load_model('D:\Documents\Thesis\Experimental_Models\Part2_FSLR_CNN_Model(30-epochs)-accuracy_0.91-val_accuracy_0.91-loss_0.27.h5')


def predict(img_arr, interval):
    temp_sentence = []
    temp_score = []
    temp_crop_img = []
    index = 0
    while index < len(img_arr):
        try:
            crop_img, roi = utils.preprocess_image(img_arr[index])
            predictions, top_predictions = SCM.classify_image(roi, model)
            score = top_predictions[4][1]
            if score >= THRESHOLD:
                word = top_predictions[4][0]
                temp_sentence.append(word)
                temp_score.append(score)
                temp_crop_img.append(crop_img)
        except Exception as exc:
            pass
        index += interval

    most_occuring_word = max(set(temp_sentence), key=temp_sentence.count)
    frm_position = temp_sentence.index(most_occuring_word)
    frm_score = temp_score[frm_position]
    crop_img = temp_crop_img[frm_position]
    return most_occuring_word, frm_position, frm_score, crop_img



while cap.isOpened():
    _, frame = cap.read()
    # Filter lines to make it sharper and smoother
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = imutils.resize(frame, width=1000)
    height, width, channel = frame.shape
    frameCopy = frame.copy()

    if not _:
        print("Ignoring empty camera frame.")
        continue;

    cv2.putText(frame, text, (10, int(0.98 * height)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    if is_capturing:
        detected, pts_upper_left, pts_lower_right = detector.find_hands(frameCopy)
        if detected:
            roi = frameCopy[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]

            try:
                currFrame = utils.convert_to_grayscale(frameCopy)
                sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
                sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
                currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

                if frm_num != 0:
                    frm_diff = cv2.absdiff(currGradient, prevGradient)
                    frm_sum = cv2.sumElems(frm_diff)
                    frm_sum = frm_sum[0]/TEN_MILLION
                    if frm_sum < GRADIENT_THRESH_VALUE:
                        img_name = os.path.join(keyframes_path, 'keyframe_' + str(frm_num) + '.jpg')
                        cv2.imwrite(img_name, frameCopy)
                        frm_sum = 0.0

                        if prev_frm_sum != 0:
                            start_index = frm_num
                        else:
                            end_index = frm_num
                    else:
                        if prev_frm_sum == 0 and start_index < end_index:
                            keyframes_arr.append((start_index, end_index))

                    prev_frm_sum = frm_sum

                    print(frm_sum, frm_num)

                    frm_gradients.append(frm_sum)
                    frm_num_arr.append(frm_num)

                cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
                prevGradient = currGradient
                frm_arr.append(frameCopy)
                crop_frm_arr.append(roi)
                frm_num += 1
            except Exception as exc:
                pass

    cv2.imshow('Original', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('s'):
        if not is_capturing:
            try:
                shutil.rmtree(keyframes_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            os.makedirs(keyframes_path)
            os.makedirs(cropped_img_path)

            text = 'Capturing'
            is_capturing = True
    elif key == ord('e'):
        if is_capturing:
            # print(keyframes_arr)
            plt.plot(frm_num_arr, frm_gradients)
            plt.savefig(os.path.join(figures_path, 'Figures.png'), bbox_inches='tight')
            plt.show()

            prev_word = ''
            sentence = []
            for (start_frm, end_frm) in keyframes_arr:
                length = end_frm - start_frm + 1
                if length > 2:
                    if length <= 5:
                        word, frm_position, frm_score, crop_img = predict(crop_frm_arr[start_frm: end_frm + 1], 1)
                    else:
                        interval = length // 5
                        word, frm_position, frm_score, crop_img = predict(crop_frm_arr[start_frm: end_frm + 1], interval)

                    frm_position += start_frm
                    if word != prev_word:
                        img_crop_path = os.path.join(cropped_img_path, str(frm_position) + '_'
                                                     + word + '_' + str(frm_score) + '.jpg')
                        cv2.imwrite(img_crop_path, crop_img)
                        sentence.append(word)
                        prev_word = word
                    print('From frame {} to {}: {} total frames {}'.format(start_frm, end_frm, length, word))

            print(sentence)
            keyframes_arr, frm_arr, frm_num_arr, frm_gradients, sentence = [], [], [], [], []
            prevGradient = np.array([])
            prev_frm_sum, start_index, end_index, frm_num = TEN_MILLION, 0, 0, 0

            text = 'Not Capturing'
            is_capturing = False


cap.release()
cv2.destroyAllWindows()