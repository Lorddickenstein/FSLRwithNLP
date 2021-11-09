import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
from datetime import datetime
import os
import Application.FslrApplication as FK
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import mediapipe as mp
# import time
# import Application.utils as utils
# import Application.HandTrackingModule as HTM
# import Application.SignClassificationModule as SCM
# import matplotlib.pyplot as plt
# import warnings
# import imutils


def showFeed():
    ret, frame = cap.read()
    height, width, channel = frame.shape
    print(height, width)

    if ret:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (0, 255, 255))
        cv2.putText(frame, "Is Capturing? {}".format(window.is_capturing), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (255, 0, 0))

        if window.is_capturing:
            frm_arr.append(frame)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        videoImg = Image.fromarray(cv2image)
        img = ImageTk.PhotoImage(image=videoImg)
        camLabel.configure(image=img)
        camLabel.imageTk = img
        camLabel.after(100, showFeed)
    else:
        camLabel.configure(image='')


def startCapture():
    frm_arr = []
    window.is_capturing = True


def endCapture():
    if window.is_capturing:
        image_name = 'signs.avi'
        path = "E:\\thesis"
        img_path = os.path.join(path, image_name)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        cap_size = (int(cap.get(3)), int(cap.get(4)))
        out = cv2.VideoWriter(img_path, fourcc, 20.0, cap_size)

        for frm in frm_arr:
            out.write(frm)
        print('video saved')
        window.is_capturing = False


# def stopCam():
#     cap.release()
#     camBut.config(text="START CAMERA", command=startCam)
#     camLabel.config(text="OFF CAM", font=("Montserrat", 12))
#
#
# def startCam():
#     cap = cv2.VideoCapture(0)
#     width_1, height_1 = 1080, 720
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)
#     camBut.config(text="STOP CAMERA", command=stopCam)
#     camLabel.config(text="HERE", font="Montserrat")
#     showFeed()

# Radius of widget
# def round_rectangle(x1, y1, x2, y2, radius=20, **kwargs):
#     points = [x1 + radius, y1,
#               x1 + radius, y1,
#               x2 - radius, y1,
#               x2 - radius, y1,
#               x2, y1,
#               x2, y1 + radius,
#               x2, y1 + radius,
#               x2, y2 - radius,
#               x2, y2 - radius,
#               x2, y2,
#               x2 - radius, y2,
#               x2 - radius, y2,
#               x1 + radius, y2,
#               x1 + radius, y2,
#               x1, y2,
#               x1, y2 - radius,
#               x1, y2 - radius,
#               x1, y1 + radius,
#               x1, y1 + radius,
#               x1, y1]
#
#     return leftFrame.create_polygon(points, **kwargs, smooth=True)
# # startBut = round_rectangle(50, 50, 150, 100, radius=20, fill="blue")

def homePage():
    window.destroy()
    import Home


window = tk.Tk()
# tf.get_logger().setLevel('ERROR')
# warnings.simplefilter(action='ignore', category=Warning)

window.is_capturing = False
cap = cv2.VideoCapture(0)
width, height = 553, 584
cap.set(553, width)
cap.set(584, height)

frm_arr = []

window.geometry("1200x650+20+20")
window.resizable(False, False)
window.title("FSLR Translator")
window.configure(background="grey")

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

# showFeed()
window.mainloop()
cap.release()
cv2.destroyAllWindows()
