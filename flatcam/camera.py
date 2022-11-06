import cv2
import threading
import time
import numpy as np


class Camera:
    """
    Class for initializing camera and capturing pictures.
    """

    def __init__(self):
        self.index = 1
        self.thread = threading.Thread(target=self.init_cam)
        self.thread.start()
        self.frame = None

    def init_cam(self):
        """
        Setting parameters of the camera.
        Returns:

        """
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # V4L2: Framework of using camera for Linux

        fourcc = cv2.VideoWriter_fourcc(*'YUY2')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, 20)  # 曝光

        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 6400)

        # Original: 640*480, calibration: 620*500
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 宽度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 高度
        cap.set(cv2.CAP_PROP_FPS, 30)  # 帧率 帧/秒
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 1)  # 亮度

        while True:
            ret, self.frame = cap.read()

    def inverse_bayer(self, img_rgb):
        """
        Simulate picture's Bayer format by sampling
        Input and ouput are all in ndarray (should cv2.imread() the image first)
        Args:
            img_rgb:

        Returns:

        """

        ''' Create the mask (arrange in Bayer array/pattern) '''
        mask = np.zeros(img_rgb.shape)
        # Blue channel filter
        mask[:, :, 0][0::2, 0::2] = 1
        # Green channel filter
        mask[:, :, 1][0::2, 1::2] = 1
        mask[:, :, 1][1::2, 0::2] = 1
        # Red channel filter
        mask[:, :, 2][1::2, 1::2] = 1

        '''Sample/Filter the RGB file using the mask'''
        bayer = mask * img_rgb

        '''Synthesize the Bayer image'''
        img = np.max(bayer,
                     axis=2)  # Find the max value for each pixel among its three channels (axis=0, axis=1, axis=2)
        img = img.astype(np.uint8)
        bayer = bayer.astype(np.uint8)
        """ img: 1-channel bayer image. bayer: 3-channel bayer image """
        return img, bayer

    def save_pic(self, directory):
        """
        Saving the picture captured to a directory.

        Args:
            directory:

        Returns:

        """

        t_start = time.time()
        # frame_bayer_1, frame_bayer_3 = self.inverse_bayer(self.frame)
        cv2.imwrite(directory, self.frame)  # RGB
        t_end = time.time()
        print("save " + directory + " successfully! Time consumed: " + str(t_end - t_start) + "s")
        self.index += 1

    def get_pic_original(self):
        print('get original frame.')
        return self.frame

    def get_pic_processed(self, angle, clip_size, downsample_size):
        img = self.frame
        width, height = img.shape[1], img.shape[0]
        if angle:
            rotate_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle,
                                                 1)  # center pos, rotation angle, zoom ratio
            rotated_img = cv2.warpAffine(img, rotate_mat, (width, height))
        else:
            rotated_img = img

        start_x = int(height / 2 - clip_size[1] / 2)  # clip_size: (width, height)
        start_y = int(width / 2 - clip_size[0] / 2)
        end_x = start_x + clip_size[1]
        end_y = start_y + clip_size[0]

        rotated_img = rotated_img[start_x:end_x, start_y:end_y]  # select desired zone
        if downsample_size < clip_size:
            rotated_img = cv2.resize(rotated_img, downsample_size)

        return rotated_img
