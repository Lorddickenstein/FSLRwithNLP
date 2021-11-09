import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow's Debugging Infos
from keras.models import load_model
import cv2
import imutils
import shutil
import numpy as np
import Application.utils as utils
import Application.HandTrackingModule as HTM
import Application.SignClassificationModule as SCM
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
if int(datetime.now().strftime('%H')) <= 12:
    GRADIENT_THRESH_VALUE = 1.6
else:
    GRADIENT_THRESH_VALUE = 4.8

TEN_MILLION = 10000000.0
THRESHOLD = 40.0
FRAME_LIMIT = 10

# Paths and Directories
figures_path = 'Figures'
keyframes_path = 'Keyframes'
cropped_img_path = 'Keyframes\Cropped Images'

# FSLR Model
model_path = 'D:\Documents\Thesis\Experimental_Models'
# model_name = 'Part2_FSLR_CNN_Model(38-epochs)-accuracy_0.91-val_accuracy_0.91-loss_0.34-val_loss_0.33.h5'
# model = load_model(os.path.join(model_path, model_name))
model_name = 'Part2_weights(20-epochs)-accuracy_0.90-val_accuracy_0.89-loss_0.41-val_loss_0.44.hdf5'
model = SCM.load_and_compile(os.path.join(model_path, model_name))

def predict(img_arr, interval):
    temp_sentence, temp_score, temp_crop_img = [], [], []
    index = 0
    while index < len(img_arr):
        try:
            crop_img, roi = utils.preprocess_image(img_arr[index])
            predictions, top_predictions = SCM.classify_image(roi, model)
            score = top_predictions[4][1]
            if score >= THRESHOLD:
                word = top_predictions[4][0]
            else:
                raise Exception()
        except Exception as exc:
            word = ''
            score = 0.0
            crop_img = []

        temp_sentence.append(word)
        temp_score.append(score)
        temp_crop_img.append(crop_img)
        index += interval

    most_occuring_word = max(set(temp_sentence), key=temp_sentence.count)
    frm_position = temp_sentence.index(most_occuring_word)
    frm_score = temp_score[frm_position]
    crop_img = temp_crop_img[frm_position]
    return most_occuring_word, frm_position * interval, frm_score, crop_img


def start_application():
    # Open the camera
    cap = cv2.VideoCapture(0)
    detector = HTM.HandDetector()

    # Detection Variables
    keyframes_arr, crop_frm_arr, frm_arr, frm_num_arr, frm_gradients = [], [], [], [], []
    prevGradient = np.array([])
    start_index, end_index, frm_num, stable_ctr = 0, 0, 0, 0
    prev_frm_sum = TEN_MILLION

    text_is_capturing = 'Not Capturing'
    color_is_capturing = (0, 0, 153)
    is_capturing = False

    while cap.isOpened():
        _, frame = cap.read()
        # Filter lines to make it sharper and smoother
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = imutils.resize(frame, width=1000)
        height, width, channel = frame.shape
        print(width, height)
        frameCopy = frame.copy()

        if not _:
            print("Ignoring empty camera frame.")
            continue;

        cv2.putText(frame, text_is_capturing, (10, int(0.98 * height)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, color_is_capturing, 4)

        if is_capturing:
            sign_captured_pos = (int(0.65 * width), int(0.05 * height))
            text_sign_captured = 'No Hands Detected.'
            color_sign_captured = (255, 255, 0)

            detected, pts_upper_left, pts_lower_right = detector.find_hands(frameCopy)
            if detected:
                roi = frameCopy[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]

                if stable_ctr >= FRAME_LIMIT:
                    sign_captured_pos = (int(0.70 * width), int(0.05 * height))
                    text_sign_captured = 'Sign Captured.'
                    color_sign_captured = (0, 255, 0)
                else:
                    sign_captured_pos = (int(0.62 * width), int(0.05 * height))
                    text_sign_captured = 'Stabilize Your Hands.'
                    color_sign_captured = (0, 0, 255)

                cv2.rectangle(frame, pts_upper_left, pts_lower_right, color_sign_captured, 4)

                try:
                    currFrame = utils.convert_to_grayscale(frameCopy)
                    sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
                    sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
                    currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

                    if frm_num != 0:
                        frm_diff = cv2.absdiff(currGradient, prevGradient)
                        frm_sum = cv2.sumElems(frm_diff)
                        frm_sum = frm_sum[0] / TEN_MILLION
                        # print(frm_sum, frm_num)

                        if frm_sum < GRADIENT_THRESH_VALUE:
                            img_name = os.path.join(keyframes_path, 'keyframe_' + str(frm_num) + '.jpg')
                            # cv2.imwrite(img_name, frameCopy)
                            stable_ctr += 1
                            frm_sum = 0.0

                            if prev_frm_sum != 0:
                                start_index = frm_num
                            else:
                                end_index = frm_num
                        else:
                            if prev_frm_sum == 0 and start_index < end_index:
                                keyframes_arr.append((start_index, end_index))
                            stable_ctr = 0

                        prev_frm_sum = frm_sum
                        print(frm_sum, frm_num)

                        frm_gradients.append(frm_sum)
                        frm_num_arr.append(frm_num)

                    prevGradient = currGradient
                    frm_arr.append(frameCopy)
                    crop_frm_arr.append(roi)
                    frm_num += 1
                except Exception as exc:
                    pass

            cv2.putText(frame, text_sign_captured, sign_captured_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_sign_captured, 3, cv2.LINE_AA)

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

                text_is_capturing = 'Capturing'
                color_is_capturing = (0, 153, 0)
                is_capturing = True
        elif key == ord('e'):
            if is_capturing:
                if prev_frm_sum == 0 and start_index < end_index:
                    keyframes_arr.append((start_index, end_index))
                date_now = datetime.now()
                fig_name = 'Figure_' + date_now.strftime('%Y-%m-%d_%H%M%S') + '.png'
                plt.plot(frm_num_arr, frm_gradients)
                plt.title('Key Frame Extraction Using The Gradient Values')
                plt.xlabel('frame')
                plt.ylabel('gradient value')
                plt.savefig(os.path.join(figures_path, fig_name), bbox_inches='tight')
                # plt.show()

                prev_word = ''
                sentence = []
                for (start_frm, end_frm) in keyframes_arr:
                    length = end_frm - start_frm + 1
                    if length >= FRAME_LIMIT:
                        interval = length // 5
                        word, frm_position, frm_score, crop_img = predict(crop_frm_arr[start_frm: (end_frm + 1)],
                                                                          interval)
                        frm_position += start_frm
                        if word != prev_word:
                            img_crop_path = os.path.join(cropped_img_path, str(frm_position) + '_'
                                                         + word + '_' + str(frm_score) + '.jpg')
                            try:
                                cv2.imwrite(img_crop_path, crop_img)
                            except Exception as exc:
                                print('Error was found at frame {}'.format(frm_position))
                                try:
                                    crop_img, _ = utils.preprocess_image(crop_frm_arr[frm_position])
                                    cv2.imwrite(img_crop_path, crop_img)
                                except Exception as exc:
                                    print('Error saving frame again')
                            sentence.append(word)
                            prev_word = word
                        print('From frame {} to {}: {} total frames {}'.format(start_frm, end_frm, length, word))

                print(sentence)
                keyframes_arr, crop_frm_arr, frm_arr, frm_num_arr, frm_gradients = [], [], [], [], []
                prevGradient = np.array([])
                start_index, end_index, frm_num, stable_ctr = 0, 0, 0, 0
                prev_frm_sum = TEN_MILLION

                text_is_capturing = 'Not Capturing'
                color_is_capturing = (0, 0, 153)
                is_capturing = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_application()