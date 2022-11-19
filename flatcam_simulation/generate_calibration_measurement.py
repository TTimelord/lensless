import matplotlib.pyplot as plt

from simulation import FlatcamSimulation
from scipy.linalg import hadamard
import numpy as np
import cv2


sensor_size = (512, 512)
scene_ratio = 0.25
CRA = np.pi / 6
fc_sim = FlatcamSimulation(sensor_size, scene_ratio, CRA)
mask = fc_sim.make_mask(mls_length=7)
psf = fc_sim.calculate_psf(size_factor=1.5, amplitude_factor=9e-5, blur=False)

# scene = np.zeros((128, 128))
# scene[64, 64] = 1
# scene = cv2.circle(scene, (64, 64), 10, 1, -1, cv2.LINE_AA)

N = 128
H = hadamard(N)
I_vector = np.ones((N, 1))

for k in range(N):
    print('generate measurement: %d/%d' % (k+1, N))
    h_k = H[:, k]
    I_vector = np.ones((N, 1))

    # horizontal
    X_k_1 = np.outer(h_k, I_vector)
    X_k_2 = -X_k_1  # The negative image
    X_k_1[X_k_1 == -1] = 0  # Set negative entries to 0
    X_k_2[X_k_2 == -1] = 0

    # visualization
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(X_k_1)
    # ax[1].imshow(X_k_2)
    # plt.show()

    meas_1 = fc_sim.simulate_measurement(X_k_1)
    meas_2 = fc_sim.simulate_measurement(X_k_2)
    meas_1 = np.dstack([meas_1*255]*3).astype(np.uint8)
    meas_2 = np.dstack([meas_2*255]*3).astype(np.uint8)
    dir_1 = "data/captured/calibration/horizontal/" + str(k + 1) + "_1.png"
    dir_2 = "data/captured/calibration/horizontal/" + str(k + 1) + "_2.png"

    cv2.imwrite(dir_1, meas_1)
    cv2.imwrite(dir_2, meas_2)

    # vertical
    X_k_1 = np.outer(I_vector, h_k)
    X_k_2 = -X_k_1  # The negative image
    X_k_1[X_k_1 == -1] = 0  # Set negative entries to 0
    X_k_2[X_k_2 == -1] = 0

    meas_1 = fc_sim.simulate_measurement(X_k_1)
    meas_2 = fc_sim.simulate_measurement(X_k_2)
    meas_1 = np.dstack([meas_1*255]*3).astype(np.uint8)
    meas_2 = np.dstack([meas_2*255]*3).astype(np.uint8)
    dir_1 = "data/captured/calibration/vertical/" + str(k + 1) + "_1.png"
    dir_2 = "data/captured/calibration/vertical/" + str(k + 1) + "_2.png"

    cv2.imwrite(dir_1, meas_1)
    cv2.imwrite(dir_2, meas_2)

