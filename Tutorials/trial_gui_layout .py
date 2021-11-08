import cv2
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
from datetime import datetime
from tkinter import messagebox
import os


def layout():
    root.cameraLabel = Label(root, text="here", borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=0, column=0, padx=10, pady=10, columnspan=5, rowspan=10)

    root.startButton = Button(root, width=10, text="START", command=startCapture)
    root.startButton.grid(row=11, column=0, sticky=W, pady=2)

    root.endButton = Button(root, width=10, text="END", command=endCapture)
    root.endButton.grid(row=12, column=0, sticky=W, pady=2)

    root.homeButton = Button(root, width=10, text="HOME")
    root.homeButton.grid(row=13, column=0, sticky=W, pady=2)

    root.camButton = Button(root, width=10, text="STOP CAMERA", command=stopCam)
    root.camButton.grid(row=14, column=0, sticky=W, pady=2)

    bagOfWordsLabel = Label(root, text="Bag of Words:", font=('ARIAL', 12))
    bagOfWordsLabel.grid(row=11, column=1, sticky=W, pady=2)

    bagOfWordsText = Text(root, bg="Light Grey", height=1, width=15)
    bagOfWordsText.grid(row=11, column=2, sticky=W, pady=2)

    bofWordsCountLabel = Label(root, text="Count:", font=('ARIAL', 12))
    bofWordsCountLabel.grid(row=11, column=4, sticky=W, pady=2)

    bofWordsCountText = Text(root, bg="Light Grey", height=1, width=3)
    bofWordsCountText.grid(row=11, column=5, sticky=W, pady=2)

    genLanLabel = Label(root, text="Generated Language:", font=('ARIAL', 12))
    genLanLabel.grid(row=13, column=1, sticky=W, pady=2)

    genLanText = Text(root, bg="Light Grey", height=1, width=15)
    genLanText.grid(row=13, column=2, sticky=W, pady=2)

    gLCountLabel = Label(root, text="Count:", font=('ARIAL', 12))
    gLCountLabel.grid(row=13, column=4, sticky=W, pady=2)

    gLCountText = Text(root, bg="Light Grey", height=1, width=3)
    gLCountText.grid(row=13, column=5, sticky=W, pady=2)

    showFeed()


def showFeed():
    ret, frame = cap.read()
    height, width, channel = frame.shape

    if ret:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (0, 255, 255))
        cv2.putText(frame, "Is Capturing? {}".format(root.is_capturing), (20,50), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (255, 0, 0))

        if root.is_capturing:
            frm_arr.append(frame)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        videoImg = Image.fromarray(cv2image)
        img = ImageTk.PhotoImage(image=videoImg)
        root.cameraLabel.configure(image=img)
        root.cameraLabel.imageTk = img
        root.cameraLabel.after(10, showFeed)
    else:
        root.cameraLabel.configure(image='')

def startCapture():
    frm_arr = []
    root.is_capturing = True

def endCapture():
    if root.is_capturing:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        image_name = 'withering.avi'
        path = ("D:\Documents\Thesis\FSLRwithNLP\Tutorials\Images")
        img_path = os.path.join(path, image_name)
        cap_size = (int(cap.get(3)), int(cap.get(4)))
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, cap_size)
        print(frm_arr)
        for frm in frm_arr:
            out.write(frm)
        print('video saved')
        root.is_capturing = False


def stopCam():
    cap.release()
    root.camButton.config(text="START CAMERA", command=startCam)
    root.cameraLabel.config(text="OFF CAM", font=('Arial', 12))


def startCam():
    cap = cv2.VideoCapture(0)
    width_1, height_1 = 1080, 200
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)
    root.camButton.config(text="STOP CAMERA", command=stopCam)
    root.cameraLabel.config(text="")
    showFeed()


root = tk.Tk()
root.is_capturing = False
cap = cv2.VideoCapture(0)

width, height = 1000, 750
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

frm_arr = []

root.title("Ready, Set, Translate")
root.geometry("1200x850")
root.resizable(False, False)
root.configure(background="grey")

dirPath = StringVar()

layout()
mainloop()
