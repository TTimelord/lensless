from scipy.io import loadmat
import flatcam
import matplotlib.pyplot as plt
import cv2
import numpy as np

""" Load images """
# meas = cv2.imread('data/captured/test/test.png')
meas_1 = cv2.imread('data/captured/calibration/horizontal/9_1.png')
meas_2 = cv2.imread('data/captured/calibration/horizontal/9_2.png')
meas = meas_1.astype(float) - meas_2.astype(float)
meas = meas.astype(float)/255
print('meas, max: %f, min: %f.' % (np.max(meas), np.min(meas)))

""" Load calibrated matrix phil and phir """
calib = loadmat('calib.mat')  # load calibration data   导入标定矩阵8个
# calib = loadmat('../flatcam_calibdata.mat')
flatcam.clean_calib(calib)

""" Reconstruct """
lmbd = 0.009  # L2 regularization parameter
# lmbd = 100
# recon = flatcam.fcrecon(meas, calib, lmbd)
recon = flatcam.fcrecon_new(meas.copy(), calib, lmbd)
print('recon, max: %f, min: %f.' % (np.max(recon), np.min(recon)))

# using the reconstructed image to regenerate sensor measurement
rowMeans = meas.mean(axis=1, keepdims=True)
colMeans = meas.mean(axis=0, keepdims=True)
allMean = rowMeans.mean()
re_meas = np.dstack((calib['P1b'] @ recon[:, :, 0] @ calib['Q1b'].T,
                    calib['P1g'] @ recon[:, :, 1] @ calib['Q1g'].T,
                    calib['P1r'] @ recon[:, :, 2] @ calib['Q1r'].T
                     ))
re_meas = re_meas + rowMeans + colMeans - allMean
print('re_meas, max: %f, min: %f.' % (np.max(re_meas), np.min(re_meas)))

# computing error between re_meas and meas
error = np.abs(re_meas - meas)
error_norm = np.linalg.norm(error, axis=2)

""" Show images """
plt.figure()
plt.subplot(4, 4, 1)
plt.imshow(cv2.cvtColor((meas*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('measurement')
plt.subplot(4, 4, 2)
plt.imshow(cv2.cvtColor(cv2.flip((recon*255).clip(0, 255).astype(np.uint8), 1), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('reconstruction')
plt.subplot(4, 4, 3)
plt.imshow(cv2.cvtColor((re_meas*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('regenerate measurement')
plt.subplot(4, 4, 4)
plt.imshow(error_norm, vmin=0, vmax=0.5)
plt.axis('off')
plt.title('error')
plt.colorbar()

color_title = ['_r', '_g', '_b']
for i in range(3):
    plt.subplot(4, 4, 1 + 4 * (i + 1))
    plt.imshow(meas[:, :, i], cmap='gray')
    plt.axis('off')
    plt.title('measurement' + color_title[i])
    plt.subplot(4, 4, 2 + 4 * (i + 1))
    plt.imshow(cv2.flip(recon[:, :, i].clip(0, 1), 1), cmap='gray')
    plt.axis('off')
    plt.title('reconstruction' + color_title[i])
    plt.subplot(4, 4, 3 + 4 * (i + 1))
    plt.imshow(re_meas[:, :, i], cmap='gray')
    plt.axis('off')
    plt.title('regenerate measurement' + color_title[i])
    plt.subplot(4, 4, 4 + 4 * (i + 1))
    plt.imshow(error[:, :, i], vmin=0, vmax=0.5)
    plt.axis('off')
    plt.title('error' + color_title[i])
    plt.colorbar()

plt.show()

