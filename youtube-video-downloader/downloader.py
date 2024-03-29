from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter import messagebox
from pytube import YouTube

root = Tk()
root.title('Youtube Video İndirme')
root.minsize(width=420, height=100)
root.resizable(width=False, height=False)


def open_file():
    global directory
    directory = askdirectory()
    print(directory)


def download():
    if len(link.get()) == 0:
        messagebox.showerror("Link Boş", "Link Boş Olmamalı")
    else:
        YouTube(link.get()).streams.first().download(directory)
        messagebox.showinfo("Tamamlandı", "Video İndirme Başarılı")


label = Label(root, text="Youtube Video İndirme", font='Helvetica 18 bold')
label.grid(row=0, column=1)

l1 = Label(root, text="Linki Gir : ", font='Helvetica 15')
l1.grid(row=1, column=0)

link = StringVar()
e1 = Entry(root, textvariable=link, width=50, borderwidth=4)
e1.grid(row=1, column=1)

button = Button(root, text="...", width=15, command=open_file)
button.grid(row=1, column=2)

button2 = Button(root, text="İndir", width=10, command=download)
button2.grid(row=2, column=1)

root.mainloop()
