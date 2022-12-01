from scipy.io import loadmat
import flatcam
import matplotlib.pyplot as plt
import numpy as np
import sys
from flatcam import make_separable


sys.path.append('/home/rvsa/tactile_lensless/flatcam_simulation')
from simulation import FlatcamSimulation
from scipy.linalg import hadamard

""" Load calibrated matrix phil and phir """
calib = loadmat('calib.mat')  # load calibration data   导入标定矩阵8个
# calib = loadmat('../flatcam_calibdata.mat')
flatcam.clean_calib(calib)

sensor_size = (512, 512)
scene_ratio = 0.25
CRA = np.pi / 6
fc_sim = FlatcamSimulation(sensor_size, scene_ratio, CRA)
mask = fc_sim.make_mask(mls_length=7)
psf = fc_sim.calculate_psf(size_factor=1.5, amplitude_factor=9e-5, blur=False)

N = 32
H = hadamard(N)
I_vector = np.ones((N, 1))

fig, ax = plt.subplots(4, 8)
for i in range(8):
    scene = np.zeros((N, N))
    scene[i*4, 16] = 1

    h_i = H[:, i]
    I_vector = np.ones((N, 1))
    # scene = np.outer(h_i, I_vector)
    # scene[scene == 0] = -1

    # re_psf = calib['P1b'] @ scene @ calib['Q1b'].T
    re_psf = np.outer(calib['P1b'][:, i*4], calib['Q1b'][:, 16])

    psf = fc_sim.simulate_measurement(scene, visualization=False)
    psf = make_separable(psf)

    im = ax[0, i].imshow(scene, cmap='gray', vmin=0, vmax=1)
    # ax[0].set_axis_off()
    im = ax[1, i].imshow(re_psf, cmap='bwr', vmin=-0.001, vmax=0.001)
    fig.colorbar(im, ax=ax[1, i])

    im = ax[2, i].imshow(psf, cmap='bwr', vmin=-0.001, vmax=0.001)
    fig.colorbar(im, ax=ax[2, i])

    im = ax[3, i].imshow(psf - re_psf, cmap='bwr')
    fig.colorbar(im, ax=ax[3, i])
plt.show()
