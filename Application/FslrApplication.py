import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import Application.utils as utils
import Application.HandTrackingModule as HTM
import Application.SignClassificationModule as SCM
import tkinter as tk
import time
# <<<<<<< HEAD
# # from tensorflow import keras
# from keras.models import load_model
# =======
# >>>>>>> 0e907a5303e43ea01e4fb36c975053f05d68d0b8
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow's Debugging Infos
# GUI Variables
cap = cv2.VideoCapture(0)
cap.set(3, 620)
cap.set(4, 480)
window = tk.Tk()
window.geometry("1250x680+20+20")
window.resizable(False, False)
window.title("FSLR Translator")
window.configure(background="grey")

# Constants
TEN_MILLION = 10000000.0
THRESHOLD = 40.0
FRAME_LIMIT = 10
THRESH_EXTRA = 0.3

# Variables
detector = HTM.HandDetector()
window.keyframes_arr, window.crop_frm_arr, window.frm_arr, window.frm_num_arr, window.frm_gradients = [], [], [], [], []
window.prevGradient = np.array([])
window.start_index, window.end_index, window.frm_num = 0, 0, 0
window.stable_ctr, window.cTime, window.GRADIENT_THRESH_VALUE = 0, 0, 0
window.prev_frm_sum = TEN_MILLION

window.text_is_capturing = 'Not Capturing'
window.color_is_capturing = (51, 51, 255)
window.is_capturing = False
window.is_calculating = True
window.gradient_thresh_arr = []
window.pTime = datetime.now().second
window.sec = 6

# Paths and Directories
figures_path = 'E:\\test\\Figures'
keyframes_path = 'E:\\test\\keyframes'
cropped_img_path = 'E:\\test\\keyframes\\cropped_images'

# FSLR Model
model_path = 'E:\\'
# # model_name = 'Part2_FSLR_CNN_Model(38-epochs)-accuracy_0.91-val_accuracy_0.91-loss_0.34-val_loss_0.33.h5'
# # model = load_model(os.path.join(model_path, model_name))
# <<<<<<< HEAD
# =======
# # model_name = 'Part2_weights(20-epochs)-accuracy_0.90-val_accuracy_0.89-loss_0.41-val_loss_0.44.hdf5'
# >>>>>>> 0e907a5303e43ea01e4fb36c975053f05d68d0b8
model_name = 'Part_2_weights_improvements-epoch_22-acc_0.94-loss_0.22-val_accuracy_0.91-val_loss_0.52.hdf5'
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

    if ret:
        # Filter lines to make it sharper and smoother
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        height, width, channel = frame.shape
        frameCopy = frame.copy()

        if window.is_calculating is False:
            cv2.putText(frame, window.text_is_capturing, (10, int(0.98 * height)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, window.color_is_capturing, 2, cv2.LINE_AA)

            if window.is_capturing:
                sign_captured_pos = (int(0.55 * width), int(0.07 * height))
                text_sign_captured = 'No Hands Detected.'
                color_sign_captured = (255, 255, 0)

                detected, pts_upper_left, pts_lower_right = detector.find_hands(frameCopy)
                if detected:
                    roi = frameCopy[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
                    currFrame = utils.convert_to_grayscale(frameCopy)
                    sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
                    sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
                    currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

                    if window.stable_ctr >= FRAME_LIMIT:
                        sign_captured_pos = (int(0.66 * width), int(0.07 * height))
                        text_sign_captured = 'Sign Captured.'
                        color_sign_captured = (0, 255, 0)
                    else:
                        sign_captured_pos = (int(0.50 * width), int(0.07 * height))
                        text_sign_captured = 'Stabilize Your Hands.'
                        color_sign_captured = (0, 0, 255)

                    cv2.rectangle(frame, pts_upper_left, pts_lower_right, color_sign_captured, 3)

                    try:
                        if window.frm_num != 0:
                            frm_diff = cv2.absdiff(currGradient, window.prevGradient)
                            frm_sum = cv2.sumElems(frm_diff)
                            frm_sum = frm_sum[0] / TEN_MILLION
                            print('%.2f' % frm_sum, window.frm_num, window.GRADIENT_THRESH_VALUE)

                            if '%.2f' % frm_sum < window.GRADIENT_THRESH_VALUE:
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

                        window.frm_arr.append(frameCopy)
                        window.crop_frm_arr.append(roi)
                    except Exception as exc:
                        pass

                    window.prevGradient = currGradient
                    window.frm_num += 1

                cv2.putText(frame, text_sign_captured, sign_captured_pos,
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, color_sign_captured, 2, cv2.LINE_AA)
        else:
            currFrame = utils.convert_to_grayscale(frameCopy)
            sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
            sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
            currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

            if window.sec >= 3:
                frame = cv2.GaussianBlur(frame, (51, 51), 0)
                cv2.putText(frame, str(window.sec - 3), (int(width / 2) - 40, int(height / 2) - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 4, (51, 51, 255), 5, cv2.LINE_AA)
                cv2.putText(frame, 'Stand in the middle of the frame.', (20, int(0.60 * height)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, 'Try not to move.', (int(0.25 * width), int(0.68 * height)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
            else:
                if window.sec >= -1:
                    cv2.putText(frame, 'Calculating average gradient.', (75, int(0.60 * height)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(frame, str(window.sec + 1), (int(width / 2) - 40, int(height / 2) - 20),
                                cv2.FONT_HERSHEY_DUPLEX, 4, (51, 255, 51), 5, cv2.LINE_AA)

                    frm_diff = cv2.absdiff(currGradient, window.prevGradient)
                    frm_sum = cv2.sumElems(frm_diff)
                    frm_sum = frm_sum[0] / TEN_MILLION
                    print('%.2f' % frm_sum, window.frm_num)
                    window.gradient_thresh_arr.append(frm_sum)
                else:
                    window.GRADIENT_THRESH_VALUE = '%.2f' % (np.mean(window.gradient_thresh_arr) + THRESH_EXTRA)
                    print('Average Gradient Difference:', window.GRADIENT_THRESH_VALUE)

                    window.is_calculating = False
                    window.frm_num = 0
                    window.prevGradient = np.array([])
                    window.gradient_thresh_arr = []
                    time.sleep(1)

            window.cTime = datetime.now().second
            if window.cTime - window.pTime == 1:
                window.sec -= 1
            window.pTime = window.cTime

            window.prevGradient = currGradient
            window.frm_num += 1

        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

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
        window.color_is_capturing = (51, 255, 51)
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
                        print('Error was found at frame {}.'.format(frm_position))
                        word = '[unrecognized]'
                        try:
                            crop_img, _ = utils.preprocess_image(window.crop_frm_arr[frm_position])
                            cv2.imwrite(img_crop_path, crop_img)
                        except Exception as exc:
                            print('Error saving frame again. Ignoring saving.')
                            word = '[unrecognized]'
                    sentence.append(word)
                    prev_word = word
                print('From frame {} to {}: {} total frames {}'.format(start_frm, end_frm, length, word))

        print(sentence)
        window.keyframes_arr, window.crop_frm_arr, window.frm_arr, window.frm_num_arr, window.frm_gradients = [], [], [], [], []
        window.prevGradient = np.array([])
        window.start_index, window.end_index, window.frm_num, window.stable_ctr = 0, 0, 0, 0
        window.prev_frm_sum = TEN_MILLION

        window.text_is_capturing = 'Not Capturing'
        window.color_is_capturing = (51, 51, 255)
        window.is_capturing = False


def set_gradient():
    window.pTime = datetime.now().second
    window.sec = 6
    window.cTime = 0
    window.is_calculating = True


def homePage():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()
    import Home


def Generate():
    pop = tk.Tk()
    pop.wm_title("Generate")
    pop.geometry("300x100")
    labelBonus = Label(pop, text="Bag of Words", font=("Montserrat", 15, "bold"))
    labelBonus.place(x=25, y=25)


leftFrame = tk.Canvas(window, width=700, height=590, bg="#c4c4c4")
leftFrame.place(x=50, y=50)

rightFrame = tk.Canvas(window, width=425, height=600, bg="#6997F3")
rightFrame.place(x=785, y=45)


camLabel = tk.Label(leftFrame, text="here", borderwidth=3, relief="groove")
camLabel.place(x=25, y=25)

startBut = tk.Button(leftFrame, width=20, height=2, text="START", bg="#1B7B03", font=("Montserrat", 9, "bold"),
                     command=startCapture)
startBut.place(x=20, y=525)
setGradBut = tk.Button(leftFrame, width=20, height=2, text='SET THRESHOLD', bg="#c4c4c4", font=("Montserrat", 9,
                                                                                                "bold"),
                       command=set_gradient)
setGradBut.place(x=350, y=525)
endBut = tk.Button(leftFrame, width=20, height=2, text="END", bg="#E21414", font=("Montserrat", 9, "bold"),
                   command=endCapture)
endBut.place(x=185, y=525)
homeBut = tk.Button(leftFrame, width=20, height=2, text="HOME", bg="#2B449D", font=("Montserrat", 9, "bold"),
                    command=homePage)
homeBut.place(x=520, y=525)

bowFrame = tk.Canvas(rightFrame, width=385, height=250, bg="#E84747")
bowFrame.place(x=20, y=20)
genLanFrame = tk.Canvas(rightFrame, width=385, height=250, bg="#E84747")
genLanFrame.place(x=20, y=330)

bowBut = tk.Button(rightFrame, width=20, height=2, text="GENERATE", bg="#c4c4c4", font=("Montserrat", 9, "bold"), command=Generate)
bowBut.place(x=260, y=280)
bowText = tk.Text(bowFrame, width=38, height=8, bg="#FDFAFA", font="Montserrat")
bowText.place(x=15, y=45)
bowCountText = tk.Text(bowFrame, width=10, height=2, bg="#FDFAFA", font="Montserrat")
bowCountText.place(x=267, y=200)
genLanText = tk.Text(genLanFrame, width=38, height=8, bg="#FDFAFA", font="Montserrat")
genLanText.place(x=15, y=45)
genLanCountText = tk.Text(genLanFrame, width=10, height=2, bg="#FDFAFA", font="Montserrat")
genLanCountText.place(x=267, y=200)

bowLabel = tk.Label(bowFrame, text="BAG OF WORDS    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 14, "bold"))
bowLabel.place(x=15, y=10)
bowCountLabel = tk.Label(bowFrame, text="COUNT    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
bowCountLabel.place(x=170, y=205)
genLanLabel = tk.Label(genLanFrame, text="GENERATED LANGUAGE    :", bg="#E84747", fg="#FDFAFA",
                       font=("Montserrat", 14, "bold"))
genLanLabel.place(x=15, y=10)
genLanCountLabel = tk.Label(genLanFrame, text="COUNT    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
genLanCountLabel.place(x=170, y=205)

start_application()
window.mainloop()
cap.release()
cv2.destroyAllWindows()
