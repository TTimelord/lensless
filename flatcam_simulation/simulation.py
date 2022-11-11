import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import max_len_seq
from scipy.linalg import hadamard
import cv2
import cupy as cp
from cupyx.scipy.signal import convolve2d


class FlatcamSimulation:
    def __init__(self, sensor_size, scene_ratio, chief_ray_angle, ):
        self.sensor_size = sensor_size
        self.scene_ratio = scene_ratio
        self.chief_ray_angle = chief_ray_angle
        # TODO: implement chief ray zone clipping
        self.mask = None
        self.psf = None

    def make_mask(self, mls_length=8):
        mls, _ = max_len_seq(mls_length)
        mls[mls == 0] = -1
        mask = np.outer(mls, mls)
        mask[mask == -1] = 0
        self.mask = mask.astype(float)
        return self.mask

    def load_mask(self, mask: np.ndarray):
        self.mask = mask

    def calculate_psf(self, size_factor, amplitude_factor, blur=False):
        """
        Calculate point spread function.
        Args:
            size_factor: The length of pixels occupied by the projection of one mask element.
            amplitude_factor: The amplitude of sensor measurement [0, 1] / The amplitude of single point in scene [0, 1]

        Returns:

        """
        size_factor *= self.sensor_size[0] / self.mask.shape[0]
        print('size_factor:', size_factor)
        self.psf = cv2.resize(self.mask,
                              dsize=(int(size_factor * self.mask.shape[1]), int(size_factor * self.mask.shape[0])),
                              interpolation=cv2.INTER_NEAREST)
        if blur:
            self.psf = cv2.blur(self.psf, (5, 5))
        self.psf *= amplitude_factor
        print('size of psf: ', self.psf.shape)
        return self.psf
        # fig, ax = plt.subplots()
        # ax.imshow(self.psf, cmap='gray')
        # plt.show()

    def simulate_measurement(self, scene: np.ndarray, blur=False):
        scene_mat = np.zeros(self.sensor_size)
        width = int(self.sensor_size[1] * self.scene_ratio)
        height = int(self.sensor_size[0] * self.scene_ratio)
        scene = cv2.resize(scene, (width, height))
        start_x = int(0.5 * scene_mat.shape[0]) - int(0.5 * height)
        start_y = int(0.5 * scene_mat.shape[1]) - int(0.5 * width)
        scene_mat[start_x:start_x + height, start_y:start_y + width] = scene
        # using cupy to accelerate conv2d
        # plt.imshow(scene_mat, cmap='gray')
        # plt.colorbar()
        # plt.show()
        measurement = cp.asnumpy(convolve2d(cp.asarray(scene_mat), cp.asarray(self.psf), 'same'))
        measurement = cv2.flip(measurement, -1)
        if blur:
            measurement = cv2.blur(measurement, (5, 5))
        return measurement


if __name__ == '__main__':
    sensor_size = (512, 512)
    scene_ratio = 0.25
    CRA = np.pi / 6
    fc_sim = FlatcamSimulation(sensor_size, scene_ratio, CRA)
    mask = fc_sim.make_mask(mls_length=7)
    psf = fc_sim.calculate_psf(size_factor=1.8, amplitude_factor=0.0001, blur=True)

    # scene = np.zeros((128, 128))
    # scene[64, 64] = 1
    # scene = cv2.circle(scene, (64, 64), 10, 1, -1, cv2.LINE_AA)

    N = 32
    H = hadamard(N)
    H[H == -1] = 0
    I_vector = np.ones((N, 1))
    h_k = H[:, 0]
    scene = np.outer(h_k, I_vector)

    meas = fc_sim.simulate_measurement(scene)
    print('meas, max = %f, min = %f:' % (np.max(meas), np.min(meas)))
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(mask, cmap='gray')
    ax[0, 0].set_title('mask')
    ax[0, 1].imshow(psf, cmap='gray')
    ax[0, 1].set_title('psf')
    ax[1, 0].imshow(scene, cmap='gray', vmin=0, vmax=1)
    ax[1, 0].set_title('scene')
    ax[1, 1].imshow(meas, cmap='gray', vmin=0, vmax=1)
    ax[1, 1].set_title('meas')
    plt.show()
