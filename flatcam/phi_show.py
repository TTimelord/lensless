from scipy.io import loadmat
import flatcam
import matplotlib.pyplot as plt
import numpy as np

""" Plot phil and phir """

calib_2 = loadmat('calib.mat')

def clean_calib( calib ):
    # Fix any formatting issues from Matlab to Python
    calib['cSize'] = np.squeeze(calib['cSize'])
    calib['angle'] = np.squeeze(calib['angle'])

# flatcam.clean_calib(calib_1)
flatcam.clean_calib(calib_2)

U_r = calib_2['P1r']
V_r = calib_2['Q1r']
U_g = calib_2['P1g']
V_g = calib_2['Q1g']

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(U_g)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(V_g.T)
plt.axis('off')
plt.show()
