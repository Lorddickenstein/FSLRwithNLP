import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
from datetime import datetime
import os
import imutils


def showFeed():
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=815)
    height, width, channel = frame.shape
    # print(height, width)

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
        camLabel.after(10, showFeed)
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
        out = cv2.VideoWriter('signs.avi', fourcc, 20.0, cap_size)

        for frm in frm_arr:
            out.write(frm)
        print('video saved')
        window.is_capturing = False


def homePage():
    window.destroy()
    import Home


def setGThresh():
    pop = tk.Tk()
    pop.wm_title("TEST")
    pop.geometry("300x100")
    labelBonus = Label(pop, text="Set Gradient Threshold", font=("Montserrat", 15, "bold"))
    labelBonus.place(x=25, y=25)


window = tk.Tk()

window.is_capturing = False
cap = cv2.VideoCapture(0)
width, height = 553, 584
cap.set(553, width)
cap.set(584, height)

frm_arr = []

window.geometry("1300x680+20+20")
window.resizable(False, False)
window.title("FSLR Translator")
window.configure(background="grey")

leftFrame = tk.Canvas(window, width=850, height=645, bg="#c4c4c4")
leftFrame.place(x=15, y=15)

rightFrame = tk.Canvas(window, width=400, height=645, bg="#6997F3")
rightFrame.place(x=880, y=15)

camLabel = tk.Label(leftFrame, text="here", borderwidth=3, relief="groove")
camLabel.place(x=20, y=20)

startBut = tk.Button(rightFrame, width=25, height=2, text="START", bg="#1B7B03", font=("Montserrat", 9, "bold"),
                     command=startCapture)
startBut.place(x=15, y=15)
justBut = tk.Button(rightFrame, width=25, height=2, bg="#c4c4c4", font=("Montserrat", 9, "bold"), command=setGThresh)
justBut.place(x=15, y=60)
endBut = tk.Button(rightFrame, width=25, height=2, text="END", bg="#E21414", font=("Montserrat", 9, "bold"),
                   command=endCapture)
endBut.place(x=205, y=15)
homeBut = tk.Button(rightFrame, width=25, height=2, text="HOME", bg="#2B449D", font=("Montserrat", 9, "bold"),
                    command=homePage)
homeBut.place(x=205, y=60)

bowFrame = tk.Canvas(rightFrame, width=370, height=250, bg="#E84747")
bowFrame.place(x=15, y=115)
genLanFrame = tk.Canvas(rightFrame, width=370, height=250, bg="#E84747")
genLanFrame.place(x=15, y=380)

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

showFeed()
window.mainloop()
cap.release()
cv2.destroyAllWindows()
