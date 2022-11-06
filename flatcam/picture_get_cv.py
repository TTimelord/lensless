from scipy.linalg import hadamard
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from run_generation import cali_gen

""" Plot the calibration images (Hadamard matrix) using open-cv, a faster way than using plt """
T_start = time.time()

N = 32
H = hadamard(N)
I_vector = np.ones((N, 1))
pic_size = 550

""" Horizontal """
for i in range(0, N):
    T1 = time.time()
    h_k = H[:, i]
    X_k_1 = np.outer(h_k, I_vector)
    X_k_2 = -X_k_1  # The negative image
    X_k_1[X_k_1 == -1] = 0  # Set negative entries to 0
    X_k_1[X_k_1 == 1] = 255
    X_k_2[X_k_2 == -1] = 0
    X_k_2[X_k_2 == 1] = 255

    """ 
    It's important to choose the appropriate interpolation algorithm for resizing
    The method using cv2 instead of plt is much faster and have better quality
    """
    pic_1 = cv2.resize(X_k_1, (pic_size, pic_size), interpolation=cv2.INTER_AREA)
    pic_2 = cv2.resize(X_k_2, (pic_size, pic_size), interpolation=cv2.INTER_AREA)
    name_1 = "data/display/calibration/horizontal/" + str(i + 1) + "_h_1.png"
    name_2 = "data/display/calibration/horizontal/" + str(i + 1) + "_h_2.png"
    cv2.imwrite(name_1, pic_1)
    cv2.imwrite(name_2, pic_2)
    T2 = time.time()
    print(name_1 + " and " + name_2 + " is saved, time consumed: " + str(T2 - T1) + "s")

""" Vertical """
for j in range(0, N):
    T1 = time.time()
    h_k = H[:, j]
    X_k_1 = np.outer(I_vector, h_k)
    X_k_2 = -X_k_1
    X_k_1[X_k_1 == -1] = 0
    X_k_1[X_k_1 == 1] = 255
    X_k_2[X_k_2 == -1] = 0
    X_k_2[X_k_2 == 1] = 255

    pic_1 = cv2.resize(X_k_1, (pic_size, pic_size), interpolation=cv2.INTER_AREA)
    pic_2 = cv2.resize(X_k_2, (pic_size, pic_size), interpolation=cv2.INTER_AREA)
    name_1 = "data/display/calibration/vertical/" + str(j + 1) + "_v_1.png"
    name_2 = "data/display/calibration/vertical/" + str(j + 1) + "_v_2.png"
    cv2.imwrite(name_1, pic_1)
    cv2.imwrite(name_2, pic_2)
    T2 = time.time()
    print(name_1 + " and " + name_2 + " is saved, time consumed: " + str(T2 - T1) + "s")

T_end = time.time()
print("Total time consumed: " + str(T_end - T_start) + "s")

""" Generate the image for display """
cali_gen(N)
