from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.font import Font
from tkinter.messagebox import showerror

root = Tk()
root.title('Basit Not Defteri')
root.minsize(width=500, height=300)
root.resizable(width=False, height=False)


def save():
    try:
        with open('note.txt', 'w') as file:
            text = textBox.get("1.0", "end-1c")
            file.write(text)
    except IOError as error:
        showerror(error)


def client_exit():
    exit()


def open_file():
    fname = askopenfilename(filetypes=(("Text files", "*.txt"),
                                       ("HTML files", "*.html;*.htm"),
                                       ("All files", "*.*")))
    with open(fname, 'r') as file:
        textBox.insert(INSERT, file.read())
        textBox.pack()


menu = Menu(root)
root.config(menu=menu)

file = Menu(menu, tearoff=0)
file.add_command(label="Dosya Aç", command=open_file)
file.add_command(label="Çıkış", command=client_exit)
menu.add_cascade(label="Dosya", menu=file)

fnt = Font(family="Helvetica", size=16)

textBox = Text(root, font=fnt)
textBox.pack()

button = Button(root, height=1, width=10, text="Kaydet", command=save, background='yellow')
button.pack()

root.mainloop()
