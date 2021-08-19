import tkinter as tk
from tkinter import filedialog, Text, Label, Grid

root = tk.Tk()
canvas = tk.Canvas(root, width=800, height=600, bg="white")
canvas.pack()

def enter_program():
    pass

frame = tk.Frame(root, bg="#3c3c3c")
frame.place(relwidth=0.9, relheight=0.9, relx=0.05, rely=0.05)

lbl_title = Label(frame, text="Ready Set Translate", fg="white",
                  bg="#3c3c3c", pady=80, font=("Harrington", 40, "bold"))
lbl_title.pack(pady=80)

btn_enter = tk.Button(frame, text="Start Translating", padx=5, pady=10,
                      fg="white", bg="#6b6b6b", command=enter_program)
btn_exit = tk.Button(frame, text="Exit", padx=10, pady=5, fg="white", bg="#6b6b6b")

btn_enter.pack()
btn_exit.pack()

root.mainloop()
