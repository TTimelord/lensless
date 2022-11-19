import numpy as np
from scipy.linalg import hadamard
import cv2
from flatcam import make_separable
from utils import process_img

"""
Read the image taken for calibration and generate phil and phir
"""

""" SVD decomposition"""
def SVD_getu(matrix):

    return u


def SVD_getvT(matrix):
    U, sigma, VT = np.linalg.svd(matrix)
    v = VT[0, :].T * np.sqrt(sigma[0])
    print(sigma[0] / sigma[1])
    return v


""" Main method for calibrating phil using horizontal strip image """


def horizontal(N, clip_size, downsample_size, angle):
    H = hadamard(N)
    H_inverse = np.linalg.inv(H)
    height = downsample_size[1]
    U_all = np.zeros(shape=(3, height, N))
    v_sign = np.zeros((3, height))
    phil = np.zeros_like(U_all)

    for i in range(1, N + 1):
        name_1 = "data/captured/calibration/horizontal/" + str(i) + "_1.png"
        name_2 = "data/captured/calibration/horizontal/" + str(i) + "_2.png"
        matrix_1 = cv2.imread(name_1)  # positive image
        matrix_2 = cv2.imread(name_2)  # negative image
        matrix_1 = process_img(matrix_1, angle, clip_size, downsample_size)
        matrix_2 = process_img(matrix_2, angle, clip_size, downsample_size)

        matrix = matrix_1.astype(np.float) - matrix_2.astype(np.float)  # subtract two sensor images
        matrix = matrix / 255  # normalize to [0, 1]

        for j in range(3):
            mat = make_separable(matrix[:, :, j])
            U, sigma, VT = np.linalg.svd(mat)
            u = U[:, 0] * np.sqrt(sigma[0])
            v = VT[0, :].T * np.sqrt(sigma[0])
            print(sigma[0] / sigma[1])  # Test for image's separability
            if i == 1:
                v_sign[j] = v
            else:
                if np.dot(v_sign[j], v) < 0:
                    u = -u
            U_all[j, :, i - 1] = u

        print("Get %sth column of u" % i)

    """ Calculate phil for each color channel """
    for i in range(3):
        phil[i] = U_all[i].dot(H_inverse)

    print("phil is generated")

    return phil


""" Main method for calibrating phir using vertical strip image """


def vertical(N, clip_size, downsample_size, angle):
    H = hadamard(N)
    H_inverse = np.linalg.inv(H)
    width = downsample_size[0]
    V_all = np.zeros(shape=(3, width, N))
    u_sign = np.zeros((3, width))
    phir = np.zeros_like(V_all)

    for i in range(1, N + 1):
        name_1 = "data/captured/calibration/vertical/" + str(i) + "_1.png"
        name_2 = "data/captured/calibration/vertical/" + str(i) + "_2.png"
        matrix_1 = cv2.imread(name_1)
        matrix_2 = cv2.imread(name_2)
        matrix_1 = process_img(matrix_1, angle, clip_size, downsample_size)
        matrix_2 = process_img(matrix_2, angle, clip_size, downsample_size)

        matrix = matrix_1.astype(np.float) - matrix_2.astype(np.float)
        matrix = matrix / 255  # normalize to [0, 1]

        for j in range(3):
            mat = make_separable(matrix[:, :, j])
            U, sigma, VT = np.linalg.svd(mat)
            u = U[:, 0] * np.sqrt(sigma[0])
            v = VT[0, :].T * np.sqrt(sigma[0])
            print(sigma[0] / sigma[1])  # Test for image's separability
            if i == 1:
                u_sign[j] = u
            else:
                if np.dot(u_sign[j], u) < 0:
                    v = -v
            V_all[j, :, i - 1] = v

        print("Get %sth column of v" % i)

    for i in range(3):
        phir[i] = V_all[i].dot(H_inverse)

    print("phir is generated")

    return phir
