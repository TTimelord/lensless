from simulation import FlatcamSimulation
import numpy as np
import cv2


sensor_size = (512, 512)
scene_ratio = 0.25
CRA = np.pi / 6
fc_sim = FlatcamSimulation(sensor_size, scene_ratio, CRA)
mask = fc_sim.make_mask(mls_length=7)
psf = fc_sim.calculate_psf(size_factor=1.5, amplitude_factor=9e-5, blur=False)

display_dir = 'data/display/test/'
captured_dir = 'data/captured/test/'
name = 'green'

scene = cv2.imread(display_dir + name + '.png')
scene_display = cv2.resize(scene, (128, 128))
cv2.imshow('asd', scene_display)
cv2.waitKey(0)
scene = scene.astype(float)/255

meas = np.zeros(sensor_size + (3, ))
for i in range(meas.shape[-1]):
    meas[:, :, i] = fc_sim.simulate_measurement(scene[:, :, i], visualization=False)

print('measurement saved at:' + captured_dir + name)
cv2.imwrite(captured_dir + name + '.png', (meas*255).astype(np.uint8))
# with open(captured_dir + name + '.npy', 'wb') as f:
#     np.save(f, meas)


