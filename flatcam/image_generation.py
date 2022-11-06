import cv2
from tkinter import *
import numpy as np
# import win32api, win32con
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QSplashScreen
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtCore import Qt
# import display_ext

""" 
Generate the image to be displayed on external screen (that is to put target image on a black background)
The size and position of the image on the screen is controllable. 
"""

""" Return the size of main screen """
def screen_size_main():
    window = Tk()
    screen_w, screen_h = window.winfo_screenwidth(), window.winfo_screenheight()
    # screen_w, screen_h = win32api.GetSystemMetrics(win32con.SM_CXSCREEN), win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    return screen_w, screen_h

    """ Return the size of external screen """
def screen_size_ext():
    app = QApplication(sys.argv)
    s = app.screens()[1]
    screen_w, screen_h = s.size().width(), s.size().height()
    return screen_w, screen_h

    """ Generate black background image """
def black_image(screen_w, screen_h):
    black = np.zeros((screen_h, screen_w, 3), dtype=int)
    cv2.imwrite('black.png', black)
    bg = cv2.imread('black.png')
    # print(bg.shape)
    return bg

    """ Calculate the start position for merging """
def center(screen_w, screen_h, width, height):
    start_x, start_y = int((screen_w - width)/2), int((screen_h - height)/2)
    return start_x, start_y

def center2(screen_w, screen_h, width, height, l):
    start_x, start_y = int((screen_w - width) / 2), int((screen_h - height) / 2)
    return start_x - l, start_y
    
    """ Put image in the center of the black background """
def merge(start_x, start_y, width, height, bg, pic):
    for i in range(3):
        """ Try not to use for loop! It's slow """
        # for j in range(start_y, start_y + height):
        #     for k in range(start_x, start_x + width):
        #         bg[:, :, i][j, k] = pic[:, :, i][j - start_y, k - start_x]

        bg[:, :, i][start_y: start_y + height, start_x: start_x + width] = pic[:, :, i]
        # Assign pic's ith channel to the central area of of bg's corresponding chanel
    return bg

    """ Show image in full-screen in the main screen """
def show_fullscreen(img):
    out_win = "output_style_full_screen"
    cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(out_win, img)
    cv2.waitKey(0)

    """ 
    Main method for the whole process 
    in_pic and out_pic: file names, resize: in pixels i.e.(550, 550) 
    """
def image_generation(in_pic, out_pic, resize):
    screen_w, screen_h = screen_size_ext()
    # Image to be displayed
    pic = cv2.imread(in_pic)
    # pic = cv2.resize(pic, (550, 550))
    pic = cv2.resize(pic, resize)
    width, height = pic.shape[1], pic.shape[0]
    start_x, start_y = center(screen_w, screen_h, width, height)
    bg = merge(start_x, start_y, width, height, black_image(screen_w, screen_h), pic)
    cv2.imwrite(out_pic, bg)
    # show_fullscreen(bg)
    
def image_generation_2(in_pic, out_pic, resize, l):
    screen_w, screen_h = screen_size_ext()
    # Image to be displayed
    pic = cv2.imread(in_pic)
    # pic = cv2.resize(pic, (550, 550))
    pic = cv2.resize(pic, resize)
    width, height = pic.shape[1], pic.shape[0]
    start_x, start_y = center2(screen_w, screen_h, width, height, l)
    bg = merge(start_x, start_y, width, height, black_image(screen_w, screen_h), pic)
    cv2.imwrite(out_pic, bg)
    # show_fullscreen(bg)

def image_generation_3(pic, out_pic):
    screen_w, screen_h = screen_size_ext()
    # Image to be displayed
    # pic = cv2.imread(in_pic)
    # pic = cv2.resize(pic, (550, 550))
    # pic = cv2.resize(pic, resize)
    width, height = pic.shape[1], pic.shape[0]
    start_x, start_y = center(screen_w, screen_h, width, height)
    bg = merge(start_x, start_y, width, height, black_image(screen_w, screen_h), pic)
    cv2.imwrite(out_pic, bg)
    # show_fullscreen(bg)









