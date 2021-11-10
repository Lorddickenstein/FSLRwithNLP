import tkinter as tk


def translating():
    root.destroy()
    import Application.FslrApplication


def about():
    root.destroy()
    import Application.GUI.AboutPage

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
