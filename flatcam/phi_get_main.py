from scipy.io import savemat
from phi_get_new import *

""" Run phi_get and store the resulting phil and phir matrix in mat file """

N = 32  # Size of hadamard matrix in calibration, size of recontructed image
m = 1024
n = 1024  # Size of image ready to be reconstructed

angle = 0  # Rotation angle. The captured pictures are already rotated so there is no need to rotate again.

phil_get = horizontal(N, m, n, angle)
phir_get = vertical(N, m, n, angle)

cSize = [m, n]
sceneSize = [N, N]
sensorZero = 0

savemat('calib.mat',
        {'P1b': phil_get[:, :, 0], 'P1g': phil_get[:, :, 1], 'P1r': phil_get[:, :, 2], 'Q1b': phir_get[:, :, 0],
         'Q1g': phir_get[:, :, 1], 'Q1r': phir_get[:, :, 2], 'cSize': cSize, 'angle': angle, 'sceneSize': sceneSize,
         'sensorZero': sensorZero})
print("phil and phir are saved")

print("Process finished!")
