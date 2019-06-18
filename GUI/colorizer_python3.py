#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tkinter

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from keras.models import load_model
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

top = Tk()
top.title = 'Colorizer'
top.geometry('900x450')
top.rowconfigure(0, minsize=25)
top.rowconfigure(1, minsize=50)
top.rowconfigure(4, minsize=50)
top.columnconfigure(0, minsize=300)
top.columnconfigure(1, minsize=300)
top.columnconfigure(2, minsize=300)
canvas1 = Canvas(top, width=256, height=256, bd=0, bg='white')
canvas1.grid(row=3, column=0)
canvas2 = Canvas(top, width=256, height=256, bd=0, bg='white')
canvas2.grid(row=3, column=1)
canvas3 = Canvas(top, width=256, height=256, bd=0, bg='white')
canvas3.grid(row=3, column=2)

e = StringVar()
f = StringVar()
g = StringVar()

def openImg():
    File = askopenfilename(title='Open Image')
    e.set(File)
    showImg(canvas1, e)
    showGrey()

def showResult(file,canvas):
    f.set(file)
    showImg(canvas,f)

def findFakeImg():
    filename = e.get()
    s = filename
    result = re.search('colorizer/(.*).jpg', s)
    f.set('fake_result/' + result.group(1) +'.jpg')
    showImg(canvas2, f)

def showGrey():
    image = load_img(e.get())
    image = image.resize((256, 256))
    image = img_to_array(image)
    color_me = np.array(image, dtype=float)
    color_me = rgb2gray(color_me)
    color_me = gray2rgb(color_me)
    imsave("gray_image.png", color_me)
    showResult("gray_image.png",canvas2)


def showImg(canvas, file):
    load = Image.open(file.get())
    load = load.resize((256, 256))
    w, h = load.size
    load = load.resize((1 * w, 1 * h))
    imgfile = ImageTk.PhotoImage(load)
    canvas.image = imgfile  # <--- keep reference of your image
    canvas.create_image(2, 2, anchor='nw', image=imgfile)

def Predict():
    # Load Model
    # mixed
    model = load_model('mixed_300_3_3250_better_landscape.h5')

    # Get test images
    image = load_img(e.get())
    image = image.resize((256, 256))
    image = img_to_array(image)
    color_me = np.array(image, dtype=float)
    color_me = rgb2lab(1.0 / 255 * color_me)[:, :, 0]
    color_me = color_me.reshape(1, 256, 256, 1)

    # Test model
    output = model.predict(color_me)
    output = output * 128

    # Output colorizations
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = color_me[0][:, :, 0]
    cur[:, :, 1:] = output[0]
    imsave("img_result.png", lab2rgb(cur))
    showResult("img_result.png",canvas3)


submit_button = Button(top, text='Open', command=openImg)
submit_button.grid(row=1, column=0)

submit_button = Button(top, text='Colorize', command=Predict)
submit_button.grid(row=1, column=2)

labelfont1 = ('times', 20, 'bold')
l1=Label(top,text='Deep Colorizer')
l1.grid(row=0, column=0, columnspan=3, sticky=N)
l1.config(font=labelfont1)

l2=Label(top,text='Please <Open> to select an image, and then press <Colorize> ')
l2.grid(row=4, column=0, columnspan=3, sticky=S)

labelfont2 = ('times', 8)
l3=Label(top,text='Author: Jiatai Han, This is a class project for CSCE636 @ TAMU, Spring 2019')
l3.grid(row=5, column=0, columnspan=3, sticky=S)
l3.config(font=labelfont2)
l3.config(fg='grey')

l4=Label(top,text='Original Image')
l4.grid(row=2, column=0, columnspan=1, sticky=S)
l5=Label(top,text='Grey-scale Image')
l5.grid(row=2, column=1, columnspan=1, sticky=S)
l6=Label(top,text='Deep-Colorized Result')
l6.grid(row=2, column=2, columnspan=1, sticky=S)

top.mainloop()
