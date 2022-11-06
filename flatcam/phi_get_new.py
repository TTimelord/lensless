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
    U, sigma, VT = np.linalg.svd(matrix)
    u = U[:, 0] * np.sqrt(sigma[0])  # Assign the singular value
    print(sigma[0] / sigma[1])  # Test for image's separability
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
    U_b = np.zeros(shape=(height, N))
    U_g = np.zeros(shape=(height, N))
    U_r = np.zeros(shape=(height, N))

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
            matrix[:, :, j] = make_separable(matrix[:, :, j])

        U_b[:, i - 1] = SVD_getu(matrix[:, :, 0])
        U_g[:, i - 1] = SVD_getu(matrix[:, :, 1])
        U_r[:, i - 1] = SVD_getu(matrix[:, :, 2])
        print("Get %sth column of u" % i)

    """ Calculate phil for each color channel """
    phil_b = U_b.dot(H_inverse)
    phil_g = U_g.dot(H_inverse)
    phil_r = U_r.dot(H_inverse)
    print("phil is generated")

    return np.dstack([phil_b, phil_g, phil_r])


""" Main method for calibrating phir using vertical strip image """


def vertical(N, clip_size, downsample_size, angle):
    H = hadamard(N)
    H_inverse = np.linalg.inv(H)
    width = downsample_size[0]
    V_b = np.zeros(shape=(width, N))
    V_g = np.zeros(shape=(width, N))
    V_r = np.zeros(shape=(width, N))

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
            matrix[:, :, j] = make_separable(matrix[:, :, j])

        V_b[:, i - 1] = SVD_getvT(matrix[:, :, 0])
        V_g[:, i - 1] = SVD_getvT(matrix[:, :, 1])
        V_r[:, i - 1] = SVD_getvT(matrix[:, :, 2])
        print("Get %sth column of v" % i)

    phir_b = V_b.dot(H_inverse)
    phir_g = V_g.dot(H_inverse)
    phir_r = V_r.dot(H_inverse)
    print("phir is generated")

    return np.dstack([phir_b, phir_g, phir_r])
