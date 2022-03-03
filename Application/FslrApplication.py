#####################################################################
# Author: Jerson Destacamento, Joshua Cruzat, Rocella Legaspi       #
# Date: October-December                                            #
# Program Title: FslrApplication.py                                 #
# Description: Contains the whole operation of the program.         #
#              Implemented with a GUI. Captures images from a       #
#              camera and predicts the sign language using the      #
#              loaded model.                                        #
# General System Design: Main Operation, CNN Part                   #
# Data structures, Algorithms, Controls: Lists, Dictionary          #
#               Tuples, Sobel Filters, GUI                          #
# Requirements: Camera (Hardware)                                   #
#####################################################################

import os
"""Disable Tensorflow's Debugging Infos"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import HandTrackingModule as HTM
import SignClassificationModule as SCM
import time
from datetime import datetime
from NLP import Tagger
from NLP import Generator
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


# Home Menu Window Declaration
home = tk.Tk()
home.geometry("1000x630+40+40")
home.resizable(False, False)
home.title("READY, SET, TRANSLATE")
home.iconbitmap('Images/logo.ico')
home.configure(background="#CF9772")


# Homepage Design
def boxDesign1():
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=810, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=840, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=870, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=900, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=930, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=960, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=35)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=810, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=840, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=870, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=900, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=930, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=960, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=70)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=810, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=840, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=870, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=900, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=930, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=960, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=105)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=810, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=840, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=870, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=900, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=930, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=960, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=140)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=175)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=175)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=175)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=175)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=210)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=210)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=210)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=210)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=245)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=245)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=245)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=245)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=280)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=280)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=280)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=280)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=315)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=315)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=315)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=315)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=350)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=350)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=350)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=350)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=990, y=385)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1020, y=385)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1050, y=385)
    d1 = Frame(home, width=10, height=10, bg="#EBD7BD")
    d1.place(x=1080, y=385)


# Homepage Design
def lineDesign2():
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=510)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=530)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=550)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=570)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=590)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=610)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=630)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=650)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=670)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=690)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=710)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=730)
    d1 = Frame(home, width=500, height=7, bg="#EBD7BD")
    d1.place(x=35, y=750)


# Homepage Header Design
def headerDesign():
    box1 = Label(headerFrame, text=" ", width=2, height=1, bg="#EBD7BD")
    box1.config(fg="#5F211B")
    box1.place(x=15, y=10)
    box2 = Label(headerFrame, text=" ", width=2, height=1, bg="#B66E28")
    box2.config(fg="#5F211B")
    box2.place(x=45, y=10)
    box3 = Label(headerFrame, text=" ", width=2, height=1, bg="#9B531A")
    box3.config(fg="#5F211B")
    box3.place(x=75, y=10)
    box4 = Label(headerFrame, text="x", width=2, height=1, bg="#C98A6D", font=('Terminal', 18))
    box4.config(fg="#5F211B")
    box4.place(x=800, y=3)


global lab
global nextBut
global prevBut
global txt
global img


# How to Use the Program
def show_how():
    how = tk.Toplevel()
    how.geometry("1000x630+40+40")
    how.resizable(False, False)
    how.title("How to Use the Program")
    how.iconbitmap('Images/logo.ico')
    how.configure(background="#CF9772")

    def exit_btn():
        how.destroy()
        how.update()

    # Header Design
    def howHeaderDesign():
        box1 = Label(howHeader, text=" ", width=2, height=1, bg="#EBD7BD")
        box1.config(fg="#5F211B")
        box1.place(x=15, y=10)
        box2 = Label(howHeader, text=" ", width=2, height=1, bg="#B66E28")
        box2.config(fg="#5F211B")
        box2.place(x=45, y=10)
        box3 = Label(howHeader, text=" ", width=2, height=1, bg="#9B531A")
        box3.config(fg="#5F211B")
        box3.place(x=75, y=10)
        box4 = Label(howHeader, text="x", width=2, height=1, bg="#C98A6D", font=('Terminal', 18))
        box4.config(fg="#5F211B")
        box4.place(x=900, y=3)

    def forwardPage(image_no, txt_no, img_no):
        global lab
        global nextBut
        global prevBut
        global txt
        global img

        txt = Label(txtFrame, text=text_List[txt_no - 1], width=30, height=28, bg="#EBD7BD",
                    font=('Terminal', 10), justify=LEFT)
        txt.place(x=5, y=4)
        img = Label(bodyFrame, text=img_List[img_no - 1], width=20, height=2, bg="#EBD7BD",
                    font=('Terminal', 10))
        img.place(x=275, y=470)
        # lab.grid_forget()
        lab = Label(imgFrame, image=image_List[image_no - 1], width=585, height=335)
        lab.place(x=5, y=5)
        nextBut = Button(bodyFrame, width=13, height=2, text="NEXT", bg="#B66E28",
                         font=('Terminal', 10), command=lambda: forwardPage(image_no + 1, txt_no + 1, img_no + 1))
        nextBut.config(fg="#5F211B")
        nextBut.place(x=425, y=470)
        prevBut = Button(bodyFrame, width=13, height=2, text="PREVIOUS", bg="#B66E28",
                         font=('Terminal', 10), command=lambda: forwardPage(image_no - 1, txt_no - 1, img_no - 1))
        prevBut.config(fg="#5F211B")
        prevBut.place(x=160, y=470)

    image11 = Image.open("Images/start.jpg")
    image11 = image11.resize((600, 350), Image.ANTIALIAS)
    image1 = ImageTk.PhotoImage(image11)
    image12 = Image.open("Images/end.jpg")
    image12 = image12.resize((600, 350), Image.ANTIALIAS)
    image2 = ImageTk.PhotoImage(image12)
    # image2 = ImageTk.PhotoImage(Image.open("end.jpg"))
    image13 = Image.open("Images/setthresh.jpg")
    image13 = image13.resize((600, 350), Image.ANTIALIAS)
    image3 = ImageTk.PhotoImage(image13)
    # image3 = ImageTk.PhotoImage(Image.open("setthresh.jpg"))
    image14 = Image.open("Images/home.jpg")
    image14 = image14.resize((600, 350), Image.ANTIALIAS)
    image4 = ImageTk.PhotoImage(image14)
    # image4 = ImageTk.PhotoImage(Image.open("home.jpg"))
    image15 = Image.open("Images/identifiedwords.jpg")
    image15 = image15.resize((600, 350), Image.ANTIALIAS)
    image5 = ImageTk.PhotoImage(image15)
    # image5 = ImageTk.PhotoImage(Image.open("identifiedwords.jpg"))
    image16 = Image.open("Images/generatedsentence.jpg")
    image16 = image16.resize((600, 350), Image.ANTIALIAS)
    image6 = ImageTk.PhotoImage(image16)
    # image6 = ImageTk.PhotoImage(Image.open("generatedsentence.jpg"))
    image17 = Image.open("Images/cam.jpg")
    image17 = image17.resize((600, 350), Image.ANTIALIAS)
    image7 = ImageTk.PhotoImage(image17)
    # image7 = ImageTk.PhotoImage(Image.open("cam.jpg"))

    text1 = '''
Start Button 
•   This button should be 
pressed first if you want to
use the application. After
pressing the start button 
you should set the threshold
first before signing any 
gesture.
'''
    text3 = '''
Set Threshold
•   After pressing this 
button, the user will see
an instruction on the 
screen to stay still. This 
will help the application 
detect if you had any 
changes in your movement.
'''
    text2 = '''
End Button
•   After signing the
desired gestures you can 
press this button to end. 
After pressing this button
the application will 
output the recognized
gestures in the 
‘Identified Words’ text
field.
'''
    text4 = '''
Home Button
•   If you want to go back
to the home screen you 
have to press this button.
'''
    text5 = '''
Identified Words
•   The text output of the 
signed gestures you made 
will be seen here. 
'''
    text6 = '''
Generated Sentence
•   The complete English 
sentence of the words you 
signed will be seen in this 
text field.
'''
    text7 = '''
Signing Screen
•   This is the screen where
you will sign your desired 
gestures. It contains 
instructions that will aid
you in signing.
'''

    imgNo1 = "image [ 1 / 7 ]"
    imgNo2 = "image [ 2 / 7 ]"
    imgNo3 = "image [ 3 / 7 ]"
    imgNo4 = "image [ 4 / 7 ]"
    imgNo5 = "image [ 5 / 7 ]"
    imgNo6 = "image [ 6 / 7 ]"
    imgNo7 = "image [ 7 / 7 ]"

    text_List = [text1, text2, text3, text4, text5, text6, text7]
    img_List = [imgNo1, imgNo2, imgNo3, imgNo4, imgNo5, imgNo6, imgNo7]
    image_List = [image1, image2, image3, image4, image5, image6, image7]

    howBorder = tk.Frame(how, width=950, height=580, bg="#9B531A")
    howBorder.place(x=25, y=25)
    howHeader = tk.Frame(howBorder, width=942, height=40, bg="#C98A6D")
    howHeader.place(x=4, y=4)
    bodyFrame = tk.Frame(howBorder, width=942, height=528, bg="#EBD7BD")
    bodyFrame.place(x=4, y=48)
    howHeaderDesign()

    imgFrame = Frame(bodyFrame, width=600, height=350, bg="#9B531A")
    imgFrame.place(x=30, y=85)
    txtFrame = Frame(bodyFrame, width=255, height=350, bg='#9B531A')
    txtFrame.place(x=655, y=85)

    txt = Label(txtFrame, text=text1, width=30, height=28, bg="#EBD7BD",
                font=('Terminal', 11), justify=LEFT)
    txt.place(x=5, y=4)
    img = Label(bodyFrame, text=imgNo1, width=20, height=2, bg="#EBD7BD",
                font=('Terminal', 10), )
    img.place(x=275, y=470)
    lab = tk.Label(imgFrame, image=image1, width=585, height=335)
    lab.place(x=5, y=5)

    homeBut = Button(bodyFrame, width=18, height=2, text="HOME",
                     bg="#B66E28", font=('Terminal', 10), command=lambda: exit_btn())
    homeBut.config(fg="#5F211B")
    homeBut.place(x=695, y=470)
    nextBut = Button(bodyFrame, width=13, height=2, text="NEXT",
                     bg="#B66E28", font=('Terminal', 10), command=lambda: forwardPage(1, 1, 1))
    nextBut.config(fg="#5F211B")
    nextBut.place(x=425, y=470)
    prevBut = Button(bodyFrame, width=13, height=2, text="PREVIOUS",
                     bg="#B66E28", font=('Terminal', 10))
    prevBut.config(fg="#5F211B")
    prevBut.place(x=160, y=470)
    header = Label(bodyFrame, text="---------- HOW TO USE ----------", bg='#EBD7BD', fg="#5F211B",
                   font=('Terminal', 17, 'bold'))
    header.place(x=325, y=30)

    # Redirect to Homepage Button


# About the Program
def show_about():
    about = tk.Toplevel()
    about.geometry("1000x630+40+40")
    about.resizable(False, False)
    about.title("SLR Translator")
    about.iconbitmap('Images/logo.ico')
    about.configure(background="#CF9772")

    # Header Design
    def aboutHeader():
        box1 = Label(headerFrame, text=" ", width=2, height=1, bg="#EBD7BD")
        box1.config(fg="#5F211B")
        box1.place(x=15, y=10)
        box2 = Label(headerFrame, text=" ", width=2, height=1, bg="#B66E28")
        box2.config(fg="#5F211B")
        box2.place(x=45, y=10)
        box3 = Label(headerFrame, text=" ", width=2, height=1, bg="#9B531A")
        box3.config(fg="#5F211B")
        box3.place(x=75, y=10)
        box4 = Label(headerFrame, text="x", width=2, height=1, bg="#C98A6D", font=('Terminal', 18))
        box4.config(fg="#5F211B")
        box4.place(x=800, y=3)

    # Redirect to Homepage Button
    def exit_btn():
        about.destroy()
        about.update()

    # Content
    def aboutCont():
        # Create a photoimage object of the image in the path
        images = Image.open("Images/logo.png")
        images = images.resize((240, 240), Image.ANTIALIAS)
        test = ImageTk.PhotoImage(images)

        label1 = Label(image=test)
        label1.image = test

        logoframe = Frame(aboutFrame, width=250, height=250, bg="#C98A6D")
        logoframe.place(x=75, y=150)

        logolabel = Label(logoframe, image=test, bg="#C98A6D")
        logolabel.image = test
        logolabel.place(x=2, y=2)

        homeBut = Button(aboutFrame, width=20, height=2, text="HOME",
                         bg="#B66E28", font=('Terminal', 10), command=exit_btn)
        homeBut.config(fg="#5F211B")
        homeBut.place(x=625, y=460)
        header = Label(aboutFrame, text="------- ABOUT THE PROGRAM -------", bg='#EBD7BD', fg="#5F211B",
                       font=('Terminal', 17, 'bold'))
        header.place(x=275, y=50)
        message = '''
    This application was made for
the betterment of communication 
between  the deaf-mute community 
also known as the “signers” and 
the normal people also known as
the “non-signers”.  When a deaf/
mute signs a Flipino Sign 
Language it outputs to a “barok” 
sentence structure. With that 
in mind, this application aims 
to generate a complete English 
sentence understandable by a 
normal non-signing person.
'''
        body = Label(aboutFrame, text=message, width=37, height=18, bg="#EBD7BD", font=('Terminal', 13), justify=LEFT)
        body.config(fg="#5F211B")
        body.place(x=375, y=110)

    aboutbrd = Frame(about, width=850, height=580, bg="#9B531A")
    aboutbrd.place(x=75, y=25)
    headerFrame = Frame(aboutbrd, width=842, height=40, bg="#C98A6D")
    headerFrame.place(x=4, y=4)
    aboutFrame = Frame(aboutbrd, width=842, height=528, bg="#EBD7BD")
    aboutFrame.place(x=4, y=48)
    aboutHeader()
    aboutCont()


# Main Program
def show_main():
    # GUI VARIABLES
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    main = tk.Toplevel()
    main.geometry("1400x800+20+20")
    main.resizable(False, False)
    main.title("FSLR Translator")
    main.iconbitmap('Images/logo.ico')
    main.configure(bg="#CF9772")

    # CONSTANT VARIABLES
    TEN_MILLION = 10000000.0
    THRESHOLD = 20.0
    FRAME_LIMIT = 10
    THRESH_EXTRA = 0.5

    # VARIABLES
    detector = HTM.HandDetector()
    main.keyframes_arr, main.crop_frm_arr, main.frm_arr, main.frm_num_arr, main.frm_gradients = [], [], [], [], []
    main.prevGradient = np.array([])
    main.start_index, main.end_index, main.frm_num = 0, 0, 0
    main.stable_ctr, main.cTime, main.GRADIENT_THRESH_VALUE = 0, 0, 0
    main.prev_frm_sum = TEN_MILLION
    main.count = 0
    main.is_using_three_models = True

    main.text_is_capturing = 'Not Capturing'
    main.color_is_capturing = (51, 51, 255)
    main.is_capturing = False
    main.is_calculating = True
    main.gradient_thresh_arr = []
    main.pTime = datetime.now().second
    main.sec = 6

    # PATHS AND DIRECTORIES
    figures_path = 'Figures'
    keyframes_path = 'Keyframes'
    cropped_img_path = 'Keyframes\Cropped Images'

    # FSLR MODEL
    model_path = 'Models'
    model_name = 'Model_1-Epochs 34.hdf5'
    model_name2 = 'Model_2-Epochs 73.hdf5'
    model_name3 = 'Model_3-Epochs 52.hdf5'
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
            main.count = len(sentence) if sentence != [''] else 0
            # Updates the bowCountText based on the number of words from the bowText
            update_text_field(bowCountText, main.count)

            if main.is_calculating is False:
                cv2.putText(frame, main.text_is_capturing, (10, int(0.98 * height)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, main.color_is_capturing, 2, cv2.LINE_AA)
                if main.is_using_three_models:
                    cv2.putText(frame, 'Using Three Models', (420, int(0.98 * height)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 128, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Using One Model', (420, int(0.98 * height)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 128, 255), 2, cv2.LINE_AA)

                if main.is_capturing:
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
                        if main.stable_ctr >= FRAME_LIMIT:
                            sign_captured_pos = (int(0.66 * width), int(0.07 * height))
                            text_sign_captured = 'Sign Captured.'
                            color_sign_captured = (0, 255, 0)
                        else:
                            sign_captured_pos = (int(0.50 * width), int(0.07 * height))
                            text_sign_captured = 'Stabilize Your Hands.'
                            color_sign_captured = (0, 0, 255)

                        cv2.rectangle(frame, pts_upper_left, pts_lower_right, color_sign_captured, 3)

                        try:
                            if main.frm_num != 0:
                                # Calculates the gradient difference between the current frame and the previous frame
                                frm_diff = cv2.absdiff(currGradient, main.prevGradient)
                                frm_sum = cv2.sumElems(frm_diff)
                                frm_sum = frm_sum[0] / TEN_MILLION
                                # print('%.2f' % frm_sum, window.frm_num, window.GRADIENT_THRESH_VALUE)

                                if '%.2f' % frm_sum < main.GRADIENT_THRESH_VALUE:
                                    # Save images if below the gradient threshold value as key frames
                                    # img_name = os.path.join(keyframes_path, 'keyframe_' + str(main.frm_num) + '.jpg')
                                    # # Uncomment to enable saving
                                    # cv2.imwrite(img_name, frameCopy)
                                    main.stable_ctr += 1
                                    frm_sum = 0.0

                                    # Determine where the key frames start and end
                                    if main.prev_frm_sum != 0:
                                        main.start_index = main.frm_num
                                    else:
                                        main.end_index = main.frm_num
                                else:
                                    if main.prev_frm_sum == 0 and main.start_index < main.end_index:
                                        main.keyframes_arr.append((main.start_index, main.end_index))
                                    main.stable_ctr = 0

                                main.prev_frm_sum = frm_sum
                                # print(frm_sum, window.frm_num)

                                main.frm_gradients.append(frm_sum)
                                main.frm_num_arr.append(main.frm_num)

                            main.frm_arr.append(frameCopy)
                            main.crop_frm_arr.append(roi)
                        except Exception as exc:
                            pass

                        main.prevGradient = currGradient
                        main.frm_num += 1

                    cv2.putText(frame, text_sign_captured, sign_captured_pos,
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, color_sign_captured, 2, cv2.LINE_AA)
            else:
                # Calculate the average gradient value of the background for 3 seconds
                currFrame = utils.convert_to_grayscale(frameCopy)
                sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
                sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
                currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

                # Starts the countdown
                if main.sec >= 3:
                    frame = cv2.GaussianBlur(frame, (51, 51), 0)
                    cv2.putText(frame, str(main.sec - 3), (int(width / 2) - 40, int(height / 2) - 20),
                                cv2.FONT_HERSHEY_DUPLEX, 4, (51, 51, 255), 5, cv2.LINE_AA)
                    cv2.putText(frame, 'Stand in the middle of the frame.', (20, int(0.60 * height)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(frame, 'Try not to move.', (int(0.25 * width), int(0.68 * height)),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
                # Starts calculating average frame gradient
                else:
                    if main.sec >= -1:
                        cv2.putText(frame, 'Calculating average gradient.', (75, int(0.60 * height)),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (102, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(frame, str(main.sec + 1), (int(width / 2) - 40, int(height / 2) - 20),
                                    cv2.FONT_HERSHEY_DUPLEX, 4, (51, 255, 51), 5, cv2.LINE_AA)

                        frm_diff = cv2.absdiff(currGradient, main.prevGradient)
                        frm_sum = cv2.sumElems(frm_diff)
                        frm_sum = frm_sum[0] / TEN_MILLION
                        # print('%.2f' % frm_sum, window.frm_num)
                        main.gradient_thresh_arr.append(frm_sum)
                    else:
                        main.GRADIENT_THRESH_VALUE = '%.2f' % (np.mean(main.gradient_thresh_arr) + THRESH_EXTRA)
                        print('Average Gradient Difference:', main.GRADIENT_THRESH_VALUE)

                        main.is_calculating = False
                        main.frm_num = 0
                        main.prevGradient = np.array([])
                        main.gradient_thresh_arr = []
                        time.sleep(1)

                main.cTime = datetime.now().second
                if main.cTime - main.pTime == 1:
                    main.sec -= 1
                main.pTime = main.cTime

                main.prevGradient = currGradient
                main.frm_num += 1

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
        if not main.is_capturing:
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

            main.text_is_capturing = 'Capturing'
            main.color_is_capturing = (51, 255, 51)
            # Switch boolean to True
            main.is_capturing = True

    def endCapture():
        """ This function displays the predictions after the user finishes his/her signing.
            Once the End Capture Button is pressed, first, this function will start predicting on
            the captured images. Next, it will display those predictions on a particular
            text field. Then, it will call the function that generates the sentence. Finally,
            it will display the generated sentence in the GUI.

            Raises:
                Exception: If image is empty and the program could not save the image
        """
        if main.is_capturing:
            if main.prev_frm_sum == 0 and main.start_index < main.end_index:
                main.keyframes_arr.append((main.start_index, main.end_index))

            # Predict on the key frames and place it on an array of Strings
            prev_word = ''
            sentence = []
            for (start_frm, end_frm) in main.keyframes_arr:
                length = end_frm - start_frm + 1
                if length >= FRAME_LIMIT:
                    # Calculate the interval to find the 5 frames to use for predictions
                    interval = length // 5
                    word1, frm_position, frm_score, crop_img = predict(main.crop_frm_arr[start_frm: (end_frm + 1)],
                                                                       interval, model1)
                    if main.is_using_three_models:
                        word2, _, _, _ = predict(main.crop_frm_arr[start_frm: (end_frm + 1)],
                                                 interval, model2)
                        word3, _, _, _ = predict(main.crop_frm_arr[start_frm: (end_frm + 1)],
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
                                crop_img, _ = utils.preprocess_image(main.crop_frm_arr[frm_position])
                                cv2.imwrite(img_crop_path, crop_img)
                            except Exception as exc:
                                print('Error saving frame again. Ignoring saving.')
                                word = '[unrecognized]'
                        sentence.append(word)
                        prev_word = word
                    print(
                        f'From frame {start_frm} to {end_frm}: [{length}] total frames, [{word}] final word, [{frm_score}] final score')

            # Plot the gradient values and save the figure as png image
            save_figures(sentence)

            print(f'\nPredictions: {sentence} \nWord Count: {len(sentence)}')
            sentence = Tagger.tokenization(sentence)
            insert_text(bowText, sentence)
            main.count = len(sentence)
            update_text_field(bowCountText, main.count)

            # Reset values of global variables
            main.keyframes_arr, main.crop_frm_arr, main.frm_arr, main.frm_num_arr, main.frm_gradients = [], [], [], [], []
            main.prevGradient = np.array([])
            main.start_index, main.end_index, main.frm_num, main.stable_ctr = 0, 0, 0, 0
            main.prev_frm_sum = TEN_MILLION

            main.text_is_capturing = 'Not Capturing'
            main.color_is_capturing = (51, 51, 255)
            main.is_capturing = False

            # Call the function that generate the sentence
            Generate()

    def save_figures(sentence):
        """ Plot the gradient values of the whole capturing session and saves the graph as a png image"""
        date_now = datetime.now()
        fig_name = 'Figure_' + date_now.strftime('%Y-%m-%d_%H%M%S') + '.png'
        plt.plot(main.frm_num_arr, main.frm_gradients)
        sentence = ' '.join([word for word in sentence])
        plt.title(f'Key Frame Extraction of {sentence}')
        plt.xlabel('frame')
        plt.ylabel('gradient value')
        plt.savefig(os.path.join(figures_path, fig_name), bbox_inches='tight')
        plt.close()

    def set_gradient():
        """ Configures the global variables for when calculating the average gradient values manually"""
        main.pTime = datetime.now().second
        main.sec = 6
        main.cTime = 0
        main.is_calculating = True

    def Generate():
        """ Generates the sentence by calling the other nlp programs and display the generated
            sentence in the GUI.
        """
        sentence = get_text(bowText)
        try:
            sentence = Tagger.separate_words(sentence.strip())
            sentence = Tagger.tokenization(sentence)

            sentence = Generator.naturalized_sentence(sentence) if sentence != '' else sentence
        except ValueError:
            sentence = 'Sentence is unrecognized.'
        update_text_field(genLanText, sentence)
        count = 0 if sentence == 'Sentence is unrecognized.' else len(sentence.split())
        update_text_field(genLanCountText, count)
        print(f'\nGenerated Sentence: {sentence}\nWord Count: {count}')

    def switch_model_num():
        """ Switches the mode of the prediction whether to use one model or three models."""
        if not main.is_using_three_models:
            bowThreeModels.config(text='Use One Model (Fast)')
            main.is_using_three_models = True
        else:
            bowThreeModels.config(text='Use Three Models (Slow, More Accurate)')
            main.is_using_three_models = False

    # Header Design
    def headerDesign1():
        box1 = Label(headerFrame, text=" ", width=2, height=1, bg="#EBD7BD")
        box1.config(fg="#5F211B")
        box1.place(x=15, y=10)
        box2 = Label(headerFrame, text=" ", width=2, height=1, bg="#B66E28")
        box2.config(fg="#5F211B")
        box2.place(x=45, y=10)
        box3 = Label(headerFrame, text=" ", width=2, height=1, bg="#9B531A")
        box3.config(fg="#5F211B")
        box3.place(x=75, y=10)
        box4 = Label(headerFrame, text="x", width=2, height=1, bg="#C98A6D", font=('Terminal', 18))
        box4.config(fg="#5F211B")
        box4.place(x=1250, y=3)

    # Redirect to Homepage Button
    def exit_btn():
        cap.release()
        main.destroy()
        main.update()

    # Content and Frames
    mainbrd = tk.Frame(main, width=1300, height=700, bg="#9B531A")
    mainbrd.place(x=50, y=50)
    headerFrame = tk.Frame(mainbrd, width=1292, height=40, bg="#C98A6D")
    headerFrame.place(x=4, y=4)
    mainBody = tk.Frame(mainbrd, width=1292, height=648, bg="#EBD7BD")
    mainBody.place(x=4, y=48)
    headerDesign1()

    rightBorder = Frame(mainBody, width=395, height=598, bg="#9B531A")
    rightBorder.place(x=870, y=25)
    rightFrame = Frame(rightBorder, width=387, height=590, bg="#EBD7BD")
    rightFrame.place(x=4, y=4)

    leftBorder = Frame(mainBody, width=660, height=498, bg="#9B531A")
    leftBorder.place(x=100, y=25)
    leftFrame = Frame(leftBorder, width=652, height=490, bg="#EBD7BD")
    leftFrame.place(x=4, y=4)

    leftBotBorder = Frame(mainBody, width=820, height=75, bg="#9B531A")
    leftBotBorder.place(x=25, y=548)
    leftBotFrame = Frame(leftBotBorder, width=812, height=67, bg="#EBD7BD")
    leftBotFrame.place(x=4, y=4)

    camLabel = Label(leftFrame, text="here", borderwidth=3, relief="groove")
    camLabel.place(x=4, y=4)

    startBut = Button(leftBotFrame, width=20, height=2, text="START", bg="#B66E28",
                      font=("Terminal", 10), command=lambda: startCapture())
    startBut.config(fg="#EBD7BD")
    startBut.place(x=25, y=15)
    setGradBut = Button(leftBotFrame, width=20, height=2, text="SET THRESHOLD",
                        bg="#B66E28", font=("Terminal", 10),command=lambda: set_gradient())
    setGradBut.config(fg="#EBD7BD")
    setGradBut.place(x=425, y=15)
    endBut = Button(leftBotFrame, width=20, height=2, text="END", bg="#B66E28",
                    font=("Terminal", 10), command=lambda: endCapture())
    endBut.config(fg="#EBD7BD")
    endBut.place(x=225, y=15)
    homeBut = Button(leftBotFrame, width=19, height=2, text="HOME", bg="#B66E28", font=("Terminal", 10),
                     command=lambda: exit_btn())
    homeBut.config(fg="#EBD7BD")
    homeBut.place(x=625, y=15)

    bowFrame = Frame(rightFrame, width=350, height=240, bg="#C98A6D")
    bowFrame.place(x=20, y=20)
    genLanFrame = Frame(rightFrame, width=350, height=240, bg="#C98A6D")
    genLanFrame.place(x=20, y=320)

    bowThreeModels = Button(rightFrame, width=42, height=2, text="Use One Model (Fast)", bg="#B66E28",
                            font=("Terminal", 10), command=lambda: switch_model_num())
    bowThreeModels.place(x=22, y=275)
    bowThreeModels.config(fg="#EBD7BD")

    bowText = Text(bowFrame, width=40, height=10, bg="#EBD7BD", font=('Terminal', 10), state=DISABLED)
    bowText.place(x=15, y=45)
    bowText.config(fg="#5F211B")
    bowCountText = Text(bowFrame, width=10, height=3, bg="#EBD7BD", font=('Terminal', 10), state=DISABLED)
    bowCountText.place(x=250, y=185)

    genLanText = Text(genLanFrame, width=40, height=10, bg="#EBD7BD", font=('Terminal', 10), state=DISABLED)
    genLanText.place(x=15, y=45)
    genLanText.config(fg="#5F211B")
    genLanCountText = Text(genLanFrame, width=10, height=3, bg="#EBD7BD", font=('Terminal', 10), state=DISABLED)
    genLanCountText.place(x=250, y=185)

    bowLabel = Label(bowFrame, text="IDENTIFIED WORDS    :", bg="#C98A6D", fg="#5F211B", font=("Terminal", 14, "bold"))
    bowLabel.place(x=15, y=10)
    bowCountLabel = Label(bowFrame, text="COUNT   :", bg="#C98A6D", fg="#5F211B", font=("Terminal", 10))
    bowCountLabel.place(x=160, y=200)

    genLanLabel = Label(genLanFrame, text="GENERATED SENTENCE    :", bg="#C98A6D", fg="#5F211B",
                        font=("Terminal", 14, "bold"))
    genLanLabel.place(x=15, y=10)
    genLanCountLabel = Label(bowFrame, text="COUNT   :", bg="#C98A6D", fg="#5F211B", font=("Terminal", 10))
    genLanCountLabel.place(x=160, y=200)

    start_application()


# Homepage Content and Button Functions
def menuFunc():
    images = Image.open("Images/logo.png")
    images = images.resize((105, 105), Image.ANTIALIAS)
    test = ImageTk.PhotoImage(images)

    label1 = Label(image=test)
    label1.image = test

    logo = Frame(aboutFrame, width=110, height=110, bg="#EBD7BD")
    logo.place(x=375, y=140)

    logolabel = Label(logo, image=test, bg="#C98A6D")
    logolabel.image = test
    logolabel.place(x=2, y=2)

    startBut = Button(aboutFrame, width=18, height=2, text="START TRANSLATING",
                      bg="#B66E28", font=('Terminal', 10), command=lambda: show_main())
    startBut.config(fg="#5F211B")
    startBut.place(x=350, y=285)
    aboutBut = Button(aboutFrame, width=18, height=2, text="ABOUT THE APP",
                      bg="#B66E28", font=('Terminal', 10), command=lambda: show_about())
    aboutBut.config(fg="#5F211B")
    aboutBut.place(x=350, y=335)
    howBut = Button(aboutFrame, width=18, height=2, text="HOW TO USE",
                    bg="#B66E28", font=('Terminal', 10), command=lambda: show_how())
    howBut.config(fg="#5F211B")
    howBut.place(x=350, y=385)
    exitBut = Button(aboutFrame, width=18, height=2, text="EXIT",
                     bg="#B66E28", font=('Terminal', 10), command=lambda: homeexit_btn())
    exitBut.config(fg="#5F211B")
    exitBut.place(x=350, y=435)
    header = Label(aboutFrame, text="READY, SET, TRANSLATE", bg='#EBD7BD', fg="#5F211B",
                   font=('Terminal', 25, 'bold'))
    header.place(x=165, y=65)
    cred = Label(aboutFrame, text="(c) Cruzat, Joshua, Destacamento, Jerson, Legaspi, Rocella", bg='#EBD7BD',
                 fg="#5F211B",
                 font=('Terminal', 10))
    cred.place(x=200, y=500)


# Homepage Exit
def homeexit_btn():
    home.destroy()


# Homepage Border
# boxDesign1()
# lineDesign2()
border = Frame(home, width=850, height=580, bg="#9B531A")
border.place(x=75, y=25)
headerFrame = Frame(border, width=842, height=40, bg="#C98A6D")
headerFrame.place(x=4, y=4)
aboutFrame = Frame(border, width=842, height=528, bg="#EBD7BD")
aboutFrame.place(x=4, y=48)
headerDesign()
menuFunc()

home.mainloop()
