from scipy.io import savemat
from phi_get_new import *

""" Run phi_get and store the resulting phil and phir matrix in mat file """

angle = 0  # -0.48 # Rotation angle of the sensor measurement.
clip_size = (512, 512)  # Size of image clipped from the original sensor measurement: (width, height)
downsample_size = (512, 512)  # downsample after clipping: (width, height)

N = 32  # Size of hadamard matrix in calibration, size of recontructed image


phil_get = horizontal(N, clip_size, downsample_size, angle)
phir_get = vertical(N, clip_size, downsample_size, angle)

clipSize = [clip_size[1], clip_size[0]]
downsampleSize = [downsample_size[1], downsample_size[0]]
sceneSize = [N, N]
sensorZero = 0

savemat('calib.mat',
        {'P1b': phil_get[:, :, 0], 'P1g': phil_get[:, :, 1], 'P1r': phil_get[:, :, 2], 'Q1b': phir_get[:, :, 0],
         'Q1g': phir_get[:, :, 1], 'Q1r': phir_get[:, :, 2], 'clipSize': clipSize, 'downsampleSize':downsampleSize,
         'angle': angle, 'sceneSize': sceneSize, 'sensorZero': sensorZero})
print("phil and phir are saved")

print("Process finished!")
