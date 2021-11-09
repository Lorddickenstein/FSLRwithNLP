import tkinter as tk

#
# def showFeed(window, cap, frm_arr, camLabel):
#     ret, frame = cap.read()
#     h, w, channel = frame.shape
#     print(h, w)
#
#     if ret:
#         frame = cv2.flip(frame, 1)
#         cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,
#                     (0, 255, 255))
#         cv2.putText(frame, "Is Capturing? {}".format(window.is_capturing), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5,
#                     (255, 0, 0))
#
#         if window.is_capturing:
#             frm_arr.append(frame)
#
#         cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#         videoImg = Image.fromarray(cv2image)
#         img = ImageTk.PhotoImage(image=videoImg)
#         camLabel.configure(image=img)
#         camLabel.imageTk = img
#         camLabel.after(100, showFeed)
#     else:
#         camLabel.configure(image='')
#
#
# def startCapture(window, frm_arr):
#     frm_arr = []
#     window.is_capturing = True
#
#
# def endCapture(window, cap, frm_arr):
#     if window.is_capturing:
#         image_name = 'signs.avi'
#         path = "E:\\thesis"
#         img_path = os.path.join(path, image_name)
#
#         fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#         cap_size = (int(cap.get(3)), int(cap.get(4)))
#         out = cv2.VideoWriter(img_path, fourcc, 20.0, cap_size)
#
#         for frm in frm_arr:
#             out.write(frm)
#         print('video saved')
#         window.is_capturing = False
#
#
# def page2():
#     window = tk.Toplevel()
#     window.is_capturing = False
#     cap = cv2.VideoCapture(0)
#     width, height = 553, 584
#     cap.set(553, width)
#     cap.set(584, height)
#
#     frm_arr = []
#
#     window.geometry("1200x650+20+20")
#     window.resizable(False, False)
#     window.title("FSLR Translator")
#     window.configure(background="grey")
#
#     leftFrame = tk.Canvas(window, width=700, height=584, bg="#c4c4c4")
#     leftFrame.place(x=35, y=35)
#
#     rightFrame = tk.Canvas(window, width=400, height=584, bg="#6997F3")
#     rightFrame.place(x=765, y=35)
#
#     camLabel = tk.Label(leftFrame, text="here", borderwidth=3, relief="groove")
#     camLabel.place(x=30, y=30)
#     record = tk.Button(leftFrame, width=20, height=2, text="START", bg="#1B7B03", font=("Montserrat", 9, "bold"),
#                        command=startCapture)
#     record.place(x=30, y=530)
#     stop = tk.Button(leftFrame, width=20, height=2, text="END", bg="#E21414", font=("Montserrat", 9, "bold"),
#                      command=endCapture)
#     stop.place(x=195, y=530)
#     homeBut = tk.Button(leftFrame, width=20, height=2, text="HOME", bg="#2B449D", font=("Montserrat", 9, "bold"),
#                         command=window.destroy)
#     homeBut.place(x=525, y=530)
#
#     bowFrame = tk.Canvas(rightFrame, width=350, height=255, bg="#E84747")
#     bowFrame.place(x=25, y=28)
#     genLanFrame = tk.Canvas(rightFrame, width=350, height=255, bg="#E84747")
#     genLanFrame.place(x=25, y=308)
#
#     bowText = tk.Text(bowFrame, width=34, height=8, bg="#FDFAFA", font="Montserrat")
#     bowText.place(x=23, y=48)
#     bowCountText = tk.Text(bowFrame, width=10, height=2, bg="#FDFAFA", font="Montserrat")
#     bowCountText.place(x=236, y=208)
#     genLanText = tk.Text(genLanFrame, width=34, height=8, bg="#FDFAFA", font="Montserrat")
#     genLanText.place(x=23, y=48)
#     genLanCountText = tk.Text(genLanFrame, width=10, height=2, bg="#FDFAFA", font="Montserrat")
#     genLanCountText.place(x=236, y=208)
#
#     bowLabel = tk.Label(bowFrame, text="BAG OF WORDS    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
#     bowLabel.place(x=23, y=16)
#     bowCountLabel = tk.Label(bowFrame, text="COUNT    :", bg="#E84747", fg="#FDFAFA", font=("Montserrat", 12, "bold"))
#     bowCountLabel.place(x=135, y=213)
#     genLanLabel = tk.Label(genLanFrame, text="GENERATED LANGUAGE    :", bg="#E84747", fg="#FDFAFA",
#                            font=("Montserrat", 12, "bold"))
#     genLanLabel.place(x=23, y=16)
#     genLanCountLabel = tk.Label(genLanFrame, text="COUNT    :", bg="#E84747", fg="#FDFAFA",
#                                 font=("Montserrat", 12, "bold"))
#     genLanCountLabel.place(x=135, y=213)
#
#     showFeed()
#     window.mainloop()
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # def page3():


# start = tk.Tk()
# start.geometry("800x500+20+20")
# start.resizable(False, False)
# start.title("FSLR Translator")
# start.configure(background="grey")
#
# startBut = tk.Button(start, width=30, height=2, text="START TRANSLATING", bg="LIGHT GREY",
#                      font=("Montserrat", 9, "bold"), command=page1)
# startBut.place(x=285, y=270)
#
# start.mainloop()


# Home Page or the Main Menu
# root = tk.Toplevel()
def translating():
    root.destroy()
    import WorkingFormat


def about():
    root.destroy()
    import AboutPage


root = tk.Tk()
root.geometry("800x500+20+20")
root.resizable(False, False)
root.title("SLR Translator")
root.configure(background="grey")

home = tk.Canvas(root, width=750, height=450, bg="#6997F3")
home.place(x=25, y=25)

titleLabel = tk.Label(home, text="READY, SET, TRANSLATE!", bg="#6997F3", fg="#FDFAFA",
                      font=("Montserrat", 30, "bold"))
titleLabel.place(x=145, y=100)
subLabel = tk.Label(home, text="Filipino Sign Language to Text", bg="#6997F3", fg="#FDFAFA",
                    font=("Montserrat", 15, "bold"))
subLabel.place(x=250, y=170)

translate = tk.Button(home, width=30, height=2, text="START TRANSLATING", bg="LIGHT GREY",
                      font=("Montserrat", 9, "bold"), command=translating)
translate.place(x=285, y=270)
aboutBut = tk.Button(home, width=20, height=2, text="ABOUT THE APP", bg="LIGHT GREY",
                     font=("Montserrat", 9, "bold"), command=about)
aboutBut.place(x=320, y=320)
endBut = tk.Button(home, width=20, height=2, text="EXIT", bg="LIGHT GREY", font=("Montserrat", 9, "bold"),
                   command=root.destroy)
endBut.place(x=320, y=370)

root.mainloop()
