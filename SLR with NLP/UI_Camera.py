import tkinter as tk
from tkinter import filedialog, Text, Label, Grid

root = tk.Tk()
canvas = tk.Canvas(root, height=600, width=800, bg="white")
canvas.pack()

def main_menu():
    # insert functionality
    pass

def start_translating():
    # insert functionality
    pass

def stop_translating():
    # insert functionality
    pass

frame = tk.Frame(root, bg="#3c3c3c")
frame.place(relwidth=0.9, relheight=0.9, relx=0.05, rely=0.05)

frm_camera = tk.Frame(frame, width=670, height=400, bg="#cfe0f0")
frm_camera.pack(pady=10)

frm_bottom = tk.Frame(frame, bg="#3c3c3c")
frm_bottom.pack(pady=10)

frm_buttons = tk.Frame(frm_bottom, width=120, height=100, bg="#6b6b6b")
frm_buttons.grid(row=0, column=0, padx=20)
frm_translations = tk.Frame(frm_bottom, width=520, height=100, bg="#cfe0f0")
frm_translations.grid(row=0, column=1, padx=10)

btn_start = tk.Button(frm_buttons, text="Start", padx=35, fg="white", bg="#6b6b6b", command=start_translating())
btn_stop = tk.Button(frm_buttons, text="Stop", padx=35, fg="white", bg="#6b6b6b", command=start_translating())
btn_home = tk.Button(frm_buttons, text="Home", padx=30, fg="white", bg="#6b6b6b", command=start_translating())
btn_start.grid(row=0, column=0)
btn_stop.grid(row=1, column=0)
btn_home.grid(row=2, column=0)

lbl_word_bag = Label(frm_translations, text="Bag of words: ", padx=200, bg="#cfe0f0")
lbl_word_bag.grid(row=0, column=0, pady=10)

lbl_generated = Label(frm_translations, text="Generated Languages: ", padx=200, bg="#cfe0f0")
lbl_generated.grid(row=1, column=0, pady=10)

root.mainloop()
