from scipy.io import loadmat
import flatcam
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils import process_img

""" Load calibrated matrix phil and phir """
calib = loadmat('calib.mat')  # load calibration data   导入标定矩阵8个
# calib = loadmat('../flatcam_calibdata.mat')
flatcam.clean_calib(calib)

""" Load images """
# meas_1 = cv2.imread('data/captured/calibration/horizontal/9_1.png')
# meas_2 = cv2.imread('data/captured/calibration/horizontal/9_2.png')
# meas_1 = cv2.imread('data/captured/calibration/vertical/21_1.png')
# meas_2 = cv2.imread('data/captured/calibration/vertical/21_2.png')
# meas_1 = cv2.imread('data/captured/calibration/horizontal/2_1.png')
# meas_2 = cv2.imread('data/captured/calibration/horizontal/2_2.png')
# meas_1 = cv2.imread('data/captured/calibration/horizontal/1_1.png')
# meas_2 = cv2.imread('data/captured/calibration/horizontal/1_2.png')

# meas_1 = cv2.imread('data/captured/calibration/horizontal/49_1.png')
# meas_2 = cv2.imread('data/captured/calibration/horizontal/49_2.png')

# meas = cv2.imread('../flatcam_simulation/data/captured/test/green.png')
meas = cv2.imread('data/captured/test/building.png')

# meas = meas_1.astype(float) - meas_2.astype(float)
meas = meas.astype(float)/255
meas_processed = process_img(meas, calib['angle'], calib['clipSize'], calib['downsampleSize'])
print('meas, max: %f, min: %f.' % (np.max(meas_processed), np.min(meas_processed)))

# cv2.imshow('rgb', meas_1)
# cv2.imshow('gray', cv2.cvtColor(meas_1, cv2.COLOR_BGR2GRAY))
# cv2.waitKey(0)

""" Reconstruct """
# lmbd = 5e-2  # L2 regularization parameter
# lmbd = 100
# recon = flatcam.fcrecon(meas, calib, lmbd)
recon = flatcam.fcrecon(meas.copy(), calib, lmbd)
recon_max = np.max(recon)
print('recon, max: %f, min: %f.' % (np.max(recon), np.min(recon)))

# using the reconstructed image to regenerate sensor measurement
rowMeans = meas_processed.mean(axis=1, keepdims=True)
colMeans = meas_processed.mean(axis=0, keepdims=True)
allMean = rowMeans.mean()
re_meas = np.dstack((calib['P1b'] @ recon[:, :, 0] @ calib['Q1b'].T,
                    calib['P1g'] @ recon[:, :, 1] @ calib['Q1g'].T,
                    calib['P1r'] @ recon[:, :, 2] @ calib['Q1r'].T
                     ))
re_meas = re_meas + rowMeans + colMeans - allMean
print('re_meas, max: %f, min: %f.' % (np.max(re_meas), np.min(re_meas)))

# computing error between re_meas and meas
error = re_meas - meas_processed
error_norm = np.linalg.norm(error, axis=2)
error_max = np.max(error_norm)

# recon = recon.clip(0)
# recon = (recon - np.min(recon))/(np.max(recon) - np.min(recon))
# recon_max = np.max(recon)

""" Show images """
plt.figure()
plt.subplot(4, 5, 1)
plt.imshow(cv2.cvtColor((meas_processed*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('measurement')
plt.subplot(4, 5, 2)
# plt.imshow(cv2.cvtColor(cv2.flip((recon*255).clip(0, 255).astype(np.uint8), 1), cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor((recon*255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('reconstruction')
# plt.subplot(4, 5, 3)
# plt.imshow(np.zeros_like(recon))
# plt.axis('off')
# plt.title('reconstruction_uncropped')
plt.subplot(4, 5, 4)
plt.imshow(cv2.cvtColor((re_meas*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('regenerate measurement')
# plt.subplot(4, 5, 5)
# plt.imshow(error_norm, vmin=-error_max, vmax=error_max, cmap='bwr')
# plt.axis('off')
# plt.title('error')
# plt.colorbar()

color_title = ['_r', '_g', '_b']
for i in range(3):
    plt.subplot(4, 5, 1 + 5 * (i + 1))
    plt.imshow(meas_processed[:, :, 2-i], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('measurement' + color_title[i])

    plt.subplot(4, 5, 2 + 5 * (i + 1))
    # plt.imshow(cv2.flip(recon[:, :, 2-i].clip(0, 1), 1), cmap='gray', vmin=0, vmax=1)
    plt.imshow(recon[:, :, 2-i].clip(0, 1), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('reconstruction' + color_title[i])

    plt.subplot(4, 5, 3 + 5 * (i + 1))
    plt.imshow(recon[:, :, 2-i], vmin=-recon_max, vmax=recon_max, cmap='bwr')
    plt.axis('off')
    plt.title('reconstruction_uncropped' + color_title[i])
    plt.colorbar(fraction=0.05, pad=0.05)

    plt.subplot(4, 5, 4 + 5 * (i + 1))
    plt.imshow(re_meas[:, :, 2-i], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('regenerate measurement' + color_title[i])

    plt.subplot(4, 5, 5 + 5 * (i + 1))
    plt.imshow(error[:, :, 2-i], vmin=-error_max, vmax=error_max, cmap='bwr')
    plt.axis('off')
    plt.title('error' + color_title[i])
    plt.colorbar(fraction=0.05, pad=0.05)

# for i in range(3):
#     plt.subplot(4, 5, 1 + 5 * (i + 1))
#     plt.imshow(meas_processed[:, :, 2 - i], cmap='bwr')
#     plt.axis('off')
#     plt.title('measurement' + color_title[i])
#     plt.colorbar(fraction=0.05, pad=0.05)
#
#     plt.subplot(4, 5, 2 + 5 * (i + 1))
#     plt.imshow(cv2.flip(recon[:, :, 2 - i].clip(0, 1), 1), cmap='gray', vmin=0, vmax=1)
#     plt.axis('off')
#     plt.title('reconstruction' + color_title[i])
#
#     plt.subplot(4, 5, 3 + 5 * (i + 1))
#     plt.imshow(cv2.flip(recon[:, :, 2 - i], 1), vmin=-recon_max, vmax=recon_max, cmap='bwr')
#     plt.axis('off')
#     plt.title('reconstruction_uncropped' + color_title[i])
#     plt.colorbar(fraction=0.05, pad=0.05)
#
#     plt.subplot(4, 5, 4 + 5 * (i + 1))
#     plt.imshow(re_meas[:, :, 2 - i], cmap='bwr')
#     plt.axis('off')
#     plt.title('regenerate measurement' + color_title[i])
#     plt.colorbar(fraction=0.05, pad=0.05)
#
#     plt.subplot(4, 5, 5 + 5 * (i + 1))
#     plt.imshow(error[:, :, 2 - i], vmin=-error_max, vmax=error_max, cmap='bwr')
#     plt.axis('off')
#     plt.title('error' + color_title[i])
#     plt.colorbar(fraction=0.05, pad=0.05)

plt.show()

