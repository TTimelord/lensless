from scipy.io import loadmat
import flatcam
import matplotlib.pyplot as plt
import cv2
import numpy as np

""" Load images """
# meas = cv2.imread('data/captured/test/test.png')
# meas = cv2.imread('data/captured/calibration/horizontal/9_1.png')
# meas_2 = cv2.imread('data/captured/calibration/horizontal/9_2.png')
# meas = meas_1 - meas_2

meas = cv2.imread('../flatcam_simulation/data/captured/test/hall.png')

meas = meas.astype(float)/255

""" Load calibrated matrix phil and phir """
calib = loadmat('calib.mat')  # load calibration data   导入标定矩阵8个
# calib = loadmat('../flatcam_calibdata.mat')
flatcam.clean_calib(calib)

""" Reconstruct """
lmbd = 3e-4  # L2 regularization parameter
# lmbd = 100
# recon = flatcam.fcrecon(meas, calib, lmbd)
recon = flatcam.fcrecon(meas.copy(), calib, lmbd)
print('max: %f, min: %f.' % (np.max(recon), np.min(recon)))

""" Show images """
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor((meas*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('FlatCam measurement')
plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(cv2.flip((recon*255).clip(0, 255).astype(np.uint8), 1), cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor((recon*255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('FlatCam reconstruction')
plt.show()
# print(recon)

# cv2.imwrite("sb.png", recon)
