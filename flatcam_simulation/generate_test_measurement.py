from simulation import FlatcamSimulation
import numpy as np
import cv2


sensor_size = (512, 512)
scene_ratio = 0.25
CRA = np.pi / 6
fc_sim = FlatcamSimulation(sensor_size, scene_ratio, CRA)
mask = fc_sim.make_mask(mls_length=7)
psf = fc_sim.calculate_psf(size_factor=1.8, amplitude_factor=0.00009, blur=False)

display_dir = 'data/display/test/'
captured_dir = 'data/captured/test/'
name = 'hall.png'

scene = cv2.imread(display_dir + name)

meas = np.zeros(sensor_size + (3, ))
for i in range(meas.shape[-1]):
    meas[:, :, i] = fc_sim.simulate_measurement(scene[:, :, i])

print('measurement saved at:' + captured_dir + name)
cv2.imwrite(captured_dir + name, meas)

