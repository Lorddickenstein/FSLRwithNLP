#####################################################################
# Author: Jerson Destacamento, Joshua Cruzat, Rocella Legaspi       #
# Date: October-December                                            #
# Description: Contains the whole operation of the program.         #
#              Implemented with a GUI. Captures images from a       #
#              camera and predicts the sign language using the      #
#              loaded model.                                        #
# General System Design: Main Operation, CNN Part                   #
# Requirements: Camera (Hardware)                                   #
#####################################################################

import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import HandTrackingModule as HTM
import SignClassificationModule as SCM
import tkinter as tk
import time
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
from NLP import Tagger
from Application.NLP import Generator

"""Disable Tensorflow's Debugging Infos"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# GUI VARIABLES
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
window = tk.Tk()
window.geometry("1250x680+20+20")
window.resizable(False, False)
window.title("FSLR Translator")
window.configure(background="grey")

# CONSTANT VARIABLES
TEN_MILLION = 10000000.0
THRESHOLD = 20.0
FRAME_LIMIT = 10
THRESH_EXTRA = 0.5

# VARIABLES
detector = HTM.HandDetector()
window.keyframes_arr, window.crop_frm_arr, window.frm_arr, window.frm_num_arr, window.frm_gradients = [], [], [], [], []
window.prevGradient = np.array([])
window.start_index, window.end_index, window.frm_num = 0, 0, 0
window.stable_ctr, window.cTime, window.GRADIENT_THRESH_VALUE = 0, 0, 0
window.prev_frm_sum = TEN_MILLION
window.count = 0
window.is_using_three_models = True

window.text_is_capturing = 'Not Capturing'
window.color_is_capturing = (51, 51, 255)
window.is_capturing = False
window.is_calculating = True
window.gradient_thresh_arr = []
window.pTime = datetime.now().second
window.sec = 6

# PATHS AND DIRECTORIES
figures_path = 'D:\Documents\Thesis\Figures'
keyframes_path = 'D:\Documents\Thesis\Keyframes'
cropped_img_path = 'D:\Documents\Thesis\Keyframes\Cropped Images'

# FSLR MODEL
model_path = 'D:\Documents\Thesis\Experimental_Models\Best so far'
model_name = 'Model_3-Epochs 35.hdf5'
model_name2 = 'Model_2-Epochs 29.hdf5'
model_name3 = 'Model_1-Epochs 38.hdf5'
# model1 = SCM.load_and_compile(os.path.join(model_path, model_name))
# model2 = SCM.load_and_compile(os.path.join(model_path, model_name2))
# model3 = SCM.load_and_compile(os.path.join(model_path, model_name3))
# model_name = 'Model_1-Epochs 34.hdf5'
# model_name2 = 'Model_2-Epochs 73.hdf5'
# model_name3 = 'Model_3-Epochs 52.hdf5'
model1 = SCM.load_and_compile(os.path.join(model_path, model_name), 1)
model2 = SCM.load_and_compile(os.path.join(model_path, model_name2), 2)
model3 = SCM.load_and_compile(os.path.join(model_path, model_name3), 3)


def predict(img_arr, interval, model):
    """ Predicts the image using the model at a specific frame interval.
        Returns the prediction, which frame it was found, the prediction score,
        and the cropped image of the hand/s.
        Always predicts on 5 equally distributed frames and looks for the most
        number of occurring word as the final predicted sign language.
        Args:
            img_arr: Numpy Array. A specific section of the cropped main array that
                contains all the significant frames to be recognized by the model.
            interval: Integer. A calculated integer that specifies which frame the model
                will predict from the img_arr. Equally divides the img_arr into 5 frames
                 to be used for prediction.
            model: Model. The model used for predicting the cropped images. The weights
                are loaded and compiled using a specific architecture to create the model.
        Returns:
            most_occurring_word: String. The final prediction of the model. The most occurring
                predicted word from the total of 5 predictions made by the model.
            frm_position: Integer. The frame position in the cropped main array where the
                most_occurring_word is found.
            frm_score: The prediction score of the image in the particular frm_position by
                the model.
            crop_img: Numpy Array. The image of the prediction word in the particular frm_position.
        Raises:
            Exception: if the image is empty, the predicted word is empty with a score of
                zero and an empty cropped crop_img.
    """
    temp_sentence, temp_score, temp_crop_img = [], [], []
    index = 0
    while index < len(img_arr):
        try:
            crop_img, roi = utils.preprocess_image(img_arr[index])
            predictions, top_predictions = SCM.classify_image(roi, model)
            score = top_predictions[4][1]
            if score >= THRESHOLD:
                word = top_predictions[4][0]
                # print(word, score)
            else:
                print('Score is below threshold')
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
    frm_position = frm_position * interval
    return most_occuring_word, frm_position, frm_score, crop_img


def update_text_field(text_field, value):
    """ Updates any generic text field. It makes the text field editable, deletes
        and inserts new content before finally making it uneditable again.
    """
    text_field['state'] = NORMAL
    text_field.delete('1.0', END)
    text_field.insert(END, value)
    text_field['state'] = DISABLED


def delete_text(text_field):
    """ Deletes the content of any generic text field."""
    text_field['state'] = NORMAL
    text_field.delete('1.0', END)
    text_field['state'] = DISABLED


def get_text(text_field):
    """ Takes the content of a text field and returns it as a string."""
    text_field['state'] = NORMAL
    text = bowText.get('1.0', END)
    text_field['state'] = DISABLED
    return text


def insert_text(text_field, text):
    """ Inserts a new content to a text field"""
    text_field['state'] = NORMAL
    text_field.insert(END, text)
    text_field['state'] = DISABLED


def start_application():
    """ This function is called when this program runs. It displays the camera
        on the GUI. This section is where the images are captured from the camera.
        At the start of the program, gradient values are computed from each of
        the frames for 3 seconds to get the average gradient value of the frame.
        The texts that are displayed on the camera section are customized in this
        function section.
        Raises:
            Exception: if image is empty and could not be saved
    """
    ret, frame = cap.read()

    if ret:
        # Filter lines to make it sharper and smoother
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        height, width, channel = frame.shape
        frameCopy = frame.copy()

        # Counts the number of identified words from the bowText text field
        sentence = get_text(bowText)
        sentence = Tagger.separate_words(sentence.strip())
        sentence = Tagger.tokenization(sentence)
        window.count = len(sentence) if sentence != [''] else 0
        # Updates the bowCountText based on the number of words from the bowText
        update_text_field(bowCountText, window.count)

        if window.is_calculating is False:
            cv2.putText(frame, window.text_is_capturing, (10, int(0.98 * height)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, window.color_is_capturing, 2, cv2.LINE_AA)
            if window.is_using_three_models:
                cv2.putText(frame, 'Using Three Models', (420, int(0.98 * height)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 128, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Using One Model', (420, int(0.98 * height)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 128, 255), 2, cv2.LINE_AA)

            if window.is_capturing:
                sign_captured_pos = (int(0.55 * width), int(0.07 * height))
                text_sign_captured = 'No Hands Detected.'
                color_sign_captured = (255, 255, 0)

                detected, pts_upper_left, pts_lower_right = detector.find_hands(frameCopy)
                if detected:
                    roi = frameCopy[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
                    # Calculate the gradient value of the current frame using the sobel operators
                    currFrame = utils.convert_to_grayscale(frameCopy)
                    sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
                    sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
                    currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

                    # Check if stable frames are greater than frame limit to detect if sign is captured or not
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
                            # Calculates the gradient difference between the current frame and the previous frame
                            frm_diff = cv2.absdiff(currGradient, window.prevGradient)
                            frm_sum = cv2.sumElems(frm_diff)
                            frm_sum = frm_sum[0] / TEN_MILLION
                            # print('%.2f' % frm_sum, window.frm_num, window.GRADIENT_THRESH_VALUE)

                            if '%.2f' % frm_sum < window.GRADIENT_THRESH_VALUE:
                                # Save images if below the gradient threshold value as key frames
                                img_name = os.path.join(keyframes_path, 'keyframe_' + str(window.frm_num) + '.jpg')
                                cv2.imwrite(img_name, frameCopy)
                                window.stable_ctr += 1
                                frm_sum = 0.0

                                # Determine where the key frames start and end
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
            # Calculate the average gradient value of the background for 3 seconds
            currFrame = utils.convert_to_grayscale(frameCopy)
            sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
            sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
            currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

            # Starts the countdown
            if window.sec >= 3:
                frame = cv2.GaussianBlur(frame, (51, 51), 0)
                cv2.putText(frame, str(window.sec - 3), (int(width / 2) - 40, int(height / 2) - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 4, (51, 51, 255), 5, cv2.LINE_AA)
                cv2.putText(frame, 'Stand in the middle of the frame.', (20, int(0.60 * height)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, 'Try not to move.', (int(0.25 * width), int(0.68 * height)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
            # Starts calculating average frame gradient
            else:
                if window.sec >= -1:
                    cv2.putText(frame, 'Calculating average gradient.', (75, int(0.60 * height)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(frame, str(window.sec + 1), (int(width / 2) - 40, int(height / 2) - 20),
                                cv2.FONT_HERSHEY_DUPLEX, 4, (51, 255, 51), 5, cv2.LINE_AA)

                    frm_diff = cv2.absdiff(currGradient, window.prevGradient)
                    frm_sum = cv2.sumElems(frm_diff)
                    frm_sum = frm_sum[0] / TEN_MILLION
                    # print('%.2f' % frm_sum, window.frm_num)
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

        # Displays the image in the GUI
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        videoImg = Image.fromarray(cv2image)
        img = ImageTk.PhotoImage(image=videoImg)
        camLabel.configure(image=img)
        camLabel.imageTk = img
        camLabel.after(1, start_application)
    else:
        camLabel.configure(image='')


def startCapture():
    """ Switches the boolean window.is_capturing from False to True to start capturing.
        Clears all the previously saved images from their respected directories and also
        the text fields to start a new sequence of sign captures.
        Raises:
            OSError: if directory is not found and therefore could not delete the folder
    """
    if not window.is_capturing:
        # Delete the whole directory instead of individually deleting all images and create a new directory
        try:
            shutil.rmtree(keyframes_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        os.makedirs(keyframes_path)
        os.makedirs(cropped_img_path)

        # Clear the content of the bowText
        delete_text(bowText)
        update_text_field(genLanText, '')
        update_text_field(genLanCountText, 0)

        window.text_is_capturing = 'Capturing'
        window.color_is_capturing = (51, 255, 51)
        # Switch boolean to True
        window.is_capturing = True


def endCapture():
    """ This function displays the predictions after the user finishes his/her signing.
        Once the End Capture Button is pressed, first, this function will start predicting on
        the captured images. Next, it will display those predictions on a particular
        text field. Then, it will call the function that generates the sentence. Finally,
        it will display the generated sentence in the GUI.
        Raises:
            Exception: If image is empty and the program could not save the image
    """
    if window.is_capturing:
        if window.prev_frm_sum == 0 and window.start_index < window.end_index:
            window.keyframes_arr.append((window.start_index, window.end_index))

        # Predict on the key frames and place it on an array of Strings
        prev_word = ''
        sentence = []
        for (start_frm, end_frm) in window.keyframes_arr:
            length = end_frm - start_frm + 1
            if length >= FRAME_LIMIT:
                # Calculate the interval to find the 5 frames to use for predictions
                interval = length // 5
                word1, frm_position, frm_score, crop_img = predict(window.crop_frm_arr[start_frm: (end_frm + 1)],
                                                                   interval, model1)
                if window.is_using_three_models:
                    word2, _, _, _ = predict(window.crop_frm_arr[start_frm: (end_frm + 1)],
                                             interval, model2)
                    word3, _, _, _ = predict(window.crop_frm_arr[start_frm: (end_frm + 1)],
                                             interval, model3)

                    word = [word1, word2, word3]
                    print(word)
                    word = max(set(word), key=word.count) if len(np.unique(word)) != 3 else word1
                else:
                    word = word1

                frm_position += start_frm
                if word != prev_word:
                    # Save the cropped image as key frames but only 1 out of 5 predictions
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
                print(f'From frame {start_frm} to {end_frm}: [{length}] total frames, [{word}] final word, '
                      f'[{frm_score}] final score')

        # Plot the gradient values and save the figure as png image
        save_figures(sentence)

        print(f'\nPredictions: {sentence} \nWord Count: {len(sentence)}')
        sentence = Tagger.tokenization(sentence)
        insert_text(bowText, sentence)
        window.count = len(sentence)
        update_text_field(bowCountText, window.count)

        # Reset values of global variables
        window.keyframes_arr, window.crop_frm_arr, window.frm_arr, window.frm_num_arr, window.frm_gradients = [], [], [], [], []
        window.prevGradient = np.array([])
        window.start_index, window.end_index, window.frm_num, window.stable_ctr = 0, 0, 0, 0
        window.prev_frm_sum = TEN_MILLION

        window.text_is_capturing = 'Not Capturing'
        window.color_is_capturing = (51, 51, 255)
        window.is_capturing = False

        # Call the function that generate the sentence
        Generate()


def save_figures(sentence):
    """ Plot the gradient values of the whole capturing session and saves the graph as a png image"""
    date_now = datetime.now()
    fig_name = 'Figure_' + date_now.strftime('%Y-%m-%d_%H%M%S') + '.png'
    plt.plot(window.frm_num_arr, window.frm_gradients)
    sentence = ' '.join([word for word in sentence])
    plt.title(f'Key Frame Extraction of {sentence}')
    plt.xlabel('frame')
    plt.ylabel('gradient value')
    plt.savefig(os.path.join(figures_path, fig_name), bbox_inches='tight')
    plt.close()


def set_gradient():
    """ Configures the global variables for when calculating the average gradient values manually"""
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
    """ Generates the sentence by calling the other nlp programs and display the generated
        sentence in the GUI.
    """
    sentence = get_text(bowText)
    sentence = Tagger.separate_words(sentence.strip())
    sentence = Tagger.tokenization(sentence)
    sentence = Generator.naturalized_sentence(sentence)
    update_text_field(genLanText, sentence)
    count = 0 if sentence == 'Sentence is unrecognized.' else len(sentence.split())
    update_text_field(genLanCountText, count)
    print(f'\nGenerated Sentence: {sentence}\nWord Count: {count}')


def switch_model_num():
    """ Switches the mode of the prediction whether to use one model or three models."""
    if not window.is_using_three_models:
        bowThreeModels.config(text='Use One Model (Fast)')
        window.is_using_three_models = True
    else:
        bowThreeModels.config(text='Use Three Models (Slow, More Accurate)')
        window.is_using_three_models = False


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

# bowBut = tk.Button(rightFrame, width=10, height=2, text="GENERATE", bg="#c4c4c4",
#                    font=("Montserrat", 9, "bold"), command=Generate)
# bowBut.place(x=330, y=280)

bowThreeModels = tk.Button(rightFrame, width=35, height=2, text="Use One Model (Fast)", bg="#c4c4c4",
                           font=("Montserrat", 9, "bold"), command=switch_model_num)
bowThreeModels.place(x=20, y=280)

bowText = tk.Text(bowFrame, width=38, height=8, bg="#FDFAFA", font="Montserrat", state=DISABLED)
bowText.place(x=15, y=45)
bowCountText = tk.Text(bowFrame, width=10, height=2, bg="#FDFAFA", font="Montserrat", state=DISABLED)
bowCountText.place(x=267, y=200)
genLanText = tk.Text(genLanFrame, width=38, height=8, bg="#FDFAFA", font="Montserrat", state=DISABLED)
genLanText.place(x=15, y=45)
genLanCountText = tk.Text(genLanFrame, width=10, height=2, bg="#FDFAFA", font="Montserrat", state=DISABLED)
genLanCountText.place(x=267, y=200)

bowLabel = tk.Label(bowFrame, text="IDENTIFIED WORDS    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 14, "bold"))
bowLabel.place(x=15, y=10)
bowCountLabel = tk.Label(bowFrame, text="COUNT    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
bowCountLabel.place(x=170, y=205)
genLanLabel = tk.Label(genLanFrame, text="GENERATED SENTENCE    :", bg="#E84747", fg="#FDFAFA",
                       font=("Montserrat", 14, "bold"))
genLanLabel.place(x=15, y=10)
genLanCountLabel = tk.Label(genLanFrame, text="COUNT    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
genLanCountLabel.place(x=170, y=205)

start_application()
window.mainloop()
cap.release()
cv2.destroyAllWindows()
