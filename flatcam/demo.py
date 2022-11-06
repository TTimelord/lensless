from scipy.io import loadmat
import flatcam
import matplotlib.pyplot as plt
import cv2
import numpy as np

""" Load images """
# meas = cv2.imread('data/captured/test/test.png')
meas = cv2.imread('data/captured/calibration/horizontal/9_1.png')
# meas_2 = cv2.imread('data/captured/calibration/horizontal/9_2.png')
# meas = meas_1 - meas_2


""" Load calibrated matrix phil and phir """
calib = loadmat('calib.mat')  # load calibration data   导入标定矩阵8个
# calib = loadmat('../flatcam_calibdata.mat')
flatcam.clean_calib(calib)

""" Reconstruct """
lmbd = 3e-4  # L2 regularization parameter
# lmbd = 100
# recon = flatcam.fcrecon(meas, calib, lmbd)
recon = flatcam.fcrecon_new(meas.copy(), calib, lmbd)
print('max: %f, min: %f.' % (np.max(recon), np.min(recon)))

# using the reconstructed image to regenerate sensor measurement
re_meas = np.dstack((calib['P1b'] @ recon[:, :, 0] @ calib['Q1b'].T,
                    calib['P1g'] @ recon[:, :, 0] @ calib['Q1g'].T,
                    calib['P1r'] @ recon[:, :, 0] @ calib['Q1r'].T
                     ))

""" Show images """
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(meas, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('FlatCam measurement')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(cv2.flip(recon.astype(np.uint8), 1), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('FlatCam reconstruction')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(re_meas.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('regenerate measurement')
plt.show()
# print(recon)

# cv2.imwrite("sb.png", recon)
