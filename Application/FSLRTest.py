import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow's Debugging Infos
import cv2
import imutils
import shutil
import numpy as np
import matplotlib.pyplot as plt
import Application.utils as utils
import Application.HandTrackingModule as HTM
import Application.SignClassificationModule as SCM
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
from keras.models import load_model

# GUI Variables
cap = cv2.VideoCapture(0)
window = tk.Tk()
window.geometry("1200x650+20+20")
window.resizable(False, False)
window.title("FSLR Translator")
window.configure(background="grey")

# Constants
if int(datetime.now().strftime('%H')) <= 12:
    GRADIENT_THRESH_VALUE = 1.6
else:
    GRADIENT_THRESH_VALUE = 4.8

TEN_MILLION = 10000000.0
THRESHOLD = 40.0
FRAME_LIMIT = 10

# Variables
detector = HTM.HandDetector()
window.keyframes_arr, window.crop_frm_arr, window.frm_arr, window.frm_num_arr, window.frm_gradients = [], [], [], [], []
window.prevGradient = np.array([])
window.start_index, window.end_index, window.frm_num, window.stable_ctr = 0, 0, 0, 0
window.prev_frm_sum = TEN_MILLION

window.text_is_capturing = 'Not Capturing'
window.color_is_capturing = (0, 0, 153)
window.is_capturing = False

# Paths and Directories
figures_path = 'D:\Documents\Thesis\Figures'
keyframes_path = 'D:\Documents\Thesis\Keyframes'
cropped_img_path = 'D:\Documents\Thesis\Keyframes\Cropped Images'

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
    ret, frame = cap.read()
    height, width, channel = frame.shape

    if ret:
        # Filter lines to make it sharper and smoother
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        # frame = imutils.resize(frame, width=1000)
        height, width, channel = frame.shape
        frameCopy = frame.copy()

        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (64, 64, 64), 1, cv2.LINE_AA)
        cv2.putText(frame, window.text_is_capturing, (10, int(0.98 * height)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, window.color_is_capturing, 3)

        if window.is_capturing:
            sign_captured_pos = (int(0.65 * width), int(0.05 * height))
            text_sign_captured = 'No Hands Detected.'
            color_sign_captured = (255, 255, 0)

            detected, pts_upper_left, pts_lower_right = detector.find_hands(frameCopy)
            if detected:
                roi = frameCopy[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]

                if window.stable_ctr >= FRAME_LIMIT:
                    sign_captured_pos = (int(0.70 * width), int(0.05 * height))
                    text_sign_captured = 'Sign Captured.'
                    color_sign_captured = (0, 255, 0)
                else:
                    sign_captured_pos = (int(0.62 * width), int(0.05 * height))
                    text_sign_captured = 'Stabilize Your Hands.'
                    color_sign_captured = (0, 0, 255)

                cv2.rectangle(frame, pts_upper_left, pts_lower_right, color_sign_captured, 3)

                try:
                    currFrame = utils.convert_to_grayscale(frameCopy)
                    sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
                    sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
                    currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

                    if window.frm_num != 0:
                        frm_diff = cv2.absdiff(currGradient, window.prevGradient)
                        frm_sum = cv2.sumElems(frm_diff)
                        frm_sum = frm_sum[0] / TEN_MILLION
                        print(frm_sum, window.frm_num)

                        if frm_sum < GRADIENT_THRESH_VALUE:
                            img_name = os.path.join(keyframes_path, 'keyframe_' + str(window.frm_num) + '.jpg')
                            cv2.imwrite(img_name, frameCopy)
                            window.stable_ctr += 1
                            frm_sum = 0.0

                            if window.prev_frm_sum != 0:
                                window.start_index = window.frm_num
                            else:
                                window.end_index = window.frm_num
                        else:
                            if window.prev_frm_sum == 0 and window.start_index < window.end_index:
                                window.keyframes_arr.append((window.start_index, window.end_index))
                            window.stable_ctr = 0

                        window.prev_frm_sum = frm_sum
                        # print(frm_sum, window.frm_num)

                        window.frm_gradients.append(frm_sum)
                        window.frm_num_arr.append(window.frm_num)

                    window.prevGradient = currGradient
                    window.frm_arr.append(frameCopy)
                    window.crop_frm_arr.append(roi)
                    window.frm_num += 1
                except Exception as exc:
                    pass

            cv2.putText(frame, text_sign_captured, sign_captured_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_sign_captured, 3, cv2.LINE_AA)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        videoImg = Image.fromarray(cv2image)
        img = ImageTk.PhotoImage(image=videoImg)
        camLabel.configure(image=img)
        camLabel.imageTk = img
        camLabel.after(1, start_application)
    else:
        camLabel.configure(image='')


def startCapture():
    if not window.is_capturing:
        try:
            shutil.rmtree(keyframes_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        os.makedirs(keyframes_path)
        os.makedirs(cropped_img_path)

        window.text_is_capturing = 'Capturing'
        window.color_is_capturing = (0, 153, 0)
        window.is_capturing = True
    window.is_capturing = True


def endCapture():
    if window.is_capturing:
        if window.prev_frm_sum == 0 and window.start_index < window.end_index:
            window.keyframes_arr.append((window.start_index, window.end_index))
        date_now = datetime.now()
        fig_name = 'Figure_' + date_now.strftime('%Y-%m-%d_%H%M%S') + '.png'
        plt.plot(window.frm_num_arr, window.frm_gradients)
        plt.title('Key Frame Extraction Using The Gradient Values')
        plt.xlabel('frame')
        plt.ylabel('gradient value')
        plt.savefig(os.path.join(figures_path, fig_name), bbox_inches='tight')
        # plt.show()

        prev_word = ''
        sentence = []
        for (start_frm, end_frm) in window.keyframes_arr:
            length = end_frm - start_frm + 1
            if length >= FRAME_LIMIT:
                interval = length // 5
                word, frm_position, frm_score, crop_img = predict(window.crop_frm_arr[start_frm: (end_frm + 1)],
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
                            crop_img, _ = utils.preprocess_image(window.crop_frm_arr[frm_position])
                            cv2.imwrite(img_crop_path, crop_img)
                        except Exception as exc:
                            print('Error saving frame again')
                            crop_img, _ = utils.preprocess_image(window.crop_frm_arr[frm_position+1])
                            cv2.imwrite(img_crop_path, crop_img)

                    sentence.append(word)
                    prev_word = word
                print('From frame {} to {}: {} total frames {}'.format(start_frm, end_frm, length, word))

        print(sentence)
        window.keyframes_arr, window.crop_frm_arr, window.frm_arr, window.frm_num_arr, window.frm_gradients = [], [], [], [], []
        window.prevGradient = np.array([])
        window.start_index, window.end_index, window.frm_num, window.stable_ctr = 0, 0, 0, 0
        window.prev_frm_sum = TEN_MILLION

        window.text_is_capturing = 'Not Capturing'
        window.color_is_capturing = (0, 0, 153)
        window.is_capturing = False


def homePage():
    window.destroy()
    import Application.GUI.Home


leftFrame = tk.Canvas(window, width=700, height=584, bg="#c4c4c4")
leftFrame.place(x=35, y=35)

rightFrame = tk.Canvas(window, width=400, height=584, bg="#6997F3")
rightFrame.place(x=765, y=35)

camLabel = tk.Label(leftFrame, text="here", borderwidth=3, relief="groove")
camLabel.place(x=30, y=30)
startBut = tk.Button(leftFrame, width=20, height=2, text="START", bg="#1B7B03", font=("Montserrat", 9, "bold"),
                     command=startCapture)
startBut.place(x=30, y=530)
endBut = tk.Button(leftFrame, width=20, height=2, text="END", bg="#E21414", font=("Montserrat", 9, "bold"),
                   command=endCapture)
endBut.place(x=195, y=530)
homeBut = tk.Button(leftFrame, width=20, height=2, text="HOME", bg="#2B449D", font=("Montserrat", 9, "bold"), command=homePage)
homeBut.place(x=525, y=530)

bowFrame = tk.Canvas(rightFrame, width=350, height=255, bg="#E84747")
bowFrame.place(x=25, y=28)
genLanFrame = tk.Canvas(rightFrame, width=350, height=255, bg="#E84747")
genLanFrame.place(x=25, y=308)

bowText = tk.Text(bowFrame, width=34, height=8, bg="#FDFAFA", font="Montserrat")
bowText.place(x=23, y=48)
bowCountText = tk.Text(bowFrame, width=10, height=2, bg="#FDFAFA", font="Montserrat")
bowCountText.place(x=236, y=208)
genLanText = tk.Text(genLanFrame, width=34, height=8, bg="#FDFAFA", font="Montserrat")
genLanText.place(x=23, y=48)
genLanCountText = tk.Text(genLanFrame, width=10, height=2, bg="#FDFAFA", font="Montserrat")
genLanCountText.place(x=236, y=208)

bowLabel = tk.Label(bowFrame, text="BAG OF WORDS    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
bowLabel.place(x=23, y=16)
bowCountLabel = tk.Label(bowFrame, text="COUNT    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
bowCountLabel.place(x=135, y=213)
genLanLabel = tk.Label(genLanFrame, text="GENERATED LANGUAGE    :", bg="#E84747", fg="#FDFAFA",
                       font=("Montserrat", 12, "bold"))
genLanLabel.place(x=23, y=16)
genLanCountLabel = tk.Label(genLanFrame, text="COUNT    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
genLanCountLabel.place(x=135, y=213)

start_application()
window.mainloop()
cap.release()
cv2.destroyAllWindows()
