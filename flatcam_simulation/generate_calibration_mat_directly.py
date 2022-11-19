import numpy as np
from simulation import FlatcamSimulation
from scipy.io import savemat

def make_separable(Y):
    """
    By subtracting row and col mean, convert sensor response back to separable image (rank-1)
    Args:
        Y: original img

    Returns:
        Ysep: separable img
    """
    rowMeans = Y.mean(axis=1, keepdims=True)
    colMeans = Y.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    Ysep = Y - rowMeans - colMeans + allMean
    return Ysep

sensor_size = (512, 512)
scene_ratio = 0.25
CRA = np.pi / 6
fc_sim = FlatcamSimulation(sensor_size, scene_ratio, CRA)
mask = fc_sim.make_mask(mls_length=7)
psf = fc_sim.calculate_psf(size_factor=4.0, amplitude_factor=6e-5, blur=False)

phil = np.zeros((512, 32))
phir = np.zeros((512, 32))

u_sign = np.zeros(512)
v_sign = np.zeros(512)

for i in range(32):
    scene = np.zeros((32, 32))
    scene[i, 0] = 1

    meas = fc_sim.simulate_measurement(scene, visualization=False)
    meas = make_separable(meas)
    U, sigma, VT = np.linalg.svd(meas)
    u = U[:, 0] * np.sqrt(sigma[0])  # Assign the singular value
    v = VT[0, :].T * np.sqrt(sigma[0])
    if i == 0:
        v_sign = v
    else:
        if np.dot(v, v_sign) < 0:
            u = -u
            print('negative')
    phil[:, i] = u
    print(i, sigma[0]/sigma[1])


for j in range(32):
    scene = np.zeros((32, 32))
    scene[0, j] = 1

    meas = fc_sim.simulate_measurement(scene, visualization=False)
    meas = make_separable(meas)
    U, sigma, VT = np.linalg.svd(meas)
    u = U[:, 0] * np.sqrt(sigma[0])
    v = VT[0, :].T * np.sqrt(sigma[0])
    if i == 0:
        u_sign = u
    else:
        if np.dot(u, u_sign) < 0:
            v = -v
            print('negative')
    print(j, sigma[0]/sigma[1])

    phir[:, j] = v

angle = 0  # -0.48 # Rotation angle of the sensor measurement.
clip_size = (512, 512)  # Size of image clipped from the original sensor measurement: (width, height)
downsample_size = (512, 512)  # downsample after clipping: (width, height)

clipSize = [clip_size[1], clip_size[0]]
downsampleSize = [downsample_size[1], downsample_size[0]]
sceneSize = [32, 32]

savemat('calib.mat',
        {'P1b': phil, 'P1g': phil, 'P1r': phil, 'Q1b': phir,
         'Q1g': phir, 'Q1r': phir,'clipSize': clipSize, 'downsampleSize':downsampleSize,
         'angle': angle, 'sceneSize': sceneSize})

