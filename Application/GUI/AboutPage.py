import tkinter as tk


def homePage():
    about.destroy()
    import Application.GUI.Home


about = tk.Tk()
about.geometry("650x550+20+20")
about.resizable(False, False)
about.title("SLR Translator")
about.configure(background="grey")

frame = tk.Canvas(about, width=550, height=450, bg="#6997F3")
frame.place(x=50, y=50)
subLabel = tk.Label(about, text="Filipino Sign Language to Text", bg="#6997F3", fg="#FDFAFA",
                    font=("Montserrat", 9, "bold"))
subLabel.place(x=250, y=100)
home = tk.Button(frame, width=10, height=2, text="HOME", bg="LIGHT GREY",
                 font=("Montserrat", 9, "bold"), command=homePage)
home.place(x=450, y=390)
about.mainloop()
