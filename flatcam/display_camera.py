from camera import Camera
from screen import myImageDisplayApp
import time
import numpy as np
import cv2

""" Automatically display image on external monitor and record them """


def calibration(N):
    """ calibration """
    for i in range(1, N+1):
        time.sleep(0.1)
        name = "data/display/calibration/horizontal/" + str(i) + "_1.png"
        myapp.emit_image_update(name)  # Update the image to be shown
        time.sleep(1.5)
        directory = "data/captured/calibration/horizontal/" + str(i) + "_1.png"
        cap_img = cam.get_pic_original()  # the angle is calibrated in rotation_calibration.py
        cv2.imwrite(directory, cap_img)

        time.sleep(0.1)
        name = "data/display/calibration/horizontal/" + str(i) + "_2.png"
        myapp.emit_image_update(name)  # Update the image to be shown
        time.sleep(1.5)
        directory = "data/captured/calibration/horizontal/" + str(i) + "_2.png"
        cam.save_pic(directory)
        cap_img = cam.get_pic_original()  # the angle is calibrated in rotation_calibration.py
        cv2.imwrite(directory, cap_img)

    for i in range(1, N+1):
        time.sleep(0.1)
        name = "data/display/calibration/vertical/" + str(i) + "_1.png"
        myapp.emit_image_update(name)
        time.sleep(1.5)
        directory = "data/captured/calibration/vertical/" + str(i) + "_1.png"
        cam.save_pic(directory)
        cap_img = cam.get_pic_original()  # the angle is calibrated in rotation_calibration.py
        cv2.imwrite(directory, cap_img)

        time.sleep(0.1)
        name = "data/display/calibration/vertical/" + str(i) + "_2.png"
        myapp.emit_image_update(name)
        time.sleep(1.5)
        directory = "data/captured/calibration/vertical/" + str(i) + "_2.png"
        cam.save_pic(directory)
        cap_img = cam.get_pic_original()  # the angle is calibrated in rotation_calibration.py
        cv2.imwrite(directory, cap_img)

    print("Process finished!")


def collect_dataset():
    """ data-set collection """
    load_txt = open("load.txt")
    paths_load = load_txt.readlines()
    index = 1
    for line in paths_load:
        time.sleep(0.1)
        line = line.strip('\n')
        myapp.emit_image_update(line)
        time.sleep(1.5)
        directory = line.replace('display', 'caps_1cap')
        directory = directory.replace('JPEG', 'png')
        directory = directory.replace('.', '..')
        cam.save_pic(directory)
        print("Progress: " + str(index) + "/" + str(len(paths_load)))
        index += 1
    print("Process finished!")


def rotation_test():
    """ Rotation test """
    for i in range(1, 50+1):
        name = "Calibration images/Rotate/" + str(i) + ".png"
        time.sleep(0.1)
        myapp.emit_image_update(name)  # Update the image to be shown
        time.sleep(1.5)
        directory = "Captured/Rotate/" + str(i) + ".png"
        cam.save_pic(directory)


def numbers():
    """ Images of numbers """
    for i in range(10):
        time.sleep(0.1)
        name = "num/" + str(i) + "_d.png"
        myapp.emit_image_update(name)  # Update the image to be shown
        time.sleep(1.5)
        directory = "num-captured/" + str(i) + ".png"
        cam.save_pic(directory)


def green_point():
    time.sleep(0.1)
    name = "data/display/point/green_point.png"
    myapp.emit_image_update(name)  # Update the image to be shown
    time.sleep(1.5)
    directory = "data/captured/point/green_point.png"
    cam.save_pic(directory)


def test():
    height, width = 2160, 3840
    img = np.zeros((height, width, 3), np.uint8)
    # cv2.circle(img, (int(width/2), int(height/2)), 20, (255, 255, 255), -1, cv2.LINE_AA)  # draw circle
    img[int(height/2)-500:int(height/2)+500, int(width/2)-60:int(width/2)+60, :] = 255
    name = "data/display/test/test.png"
    cv2.imwrite(name, img)
    myapp.emit_image_update(name)
    time.sleep(1.5)
    cap_img = cam.get_pic_original()
    cv2.imwrite('data/captured/test/test_origin.png', cap_img)
    cap_img = cam.get_pic_processed(-0.48, (1024, 1024), (512, 512))  # the angle is calibrated in rotation_calibration.py
    cv2.imwrite('data/captured/test/test_processed.png', cap_img)


if __name__ == "__main__":

    """ Start the thread for image display (copied from code on stackoverflow)"""
    cam = Camera()
    print("Camera activated now")
    time.sleep(1)

    myapp = myImageDisplayApp()
    print("Screen display start now")
    time.sleep(0.5)

    myapp.emit_image_update('data/start.png')
    time.sleep(1)

    calibration(32)
    # green_point()
    # test()


