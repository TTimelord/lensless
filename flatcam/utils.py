import cv2
import numpy as np


def process_img(img, angle, clip_size, downsample_size):
    width, height = img.shape[1], img.shape[0]
    if angle:
        rotate_mat = cv2.getRotationMatrix2D((width / 2, height / 2), float(angle),
                                             1)  # center pos, rotation angle, zoom ratio
        rotated_img = cv2.warpAffine(img, rotate_mat, (width, height))
    else:
        rotated_img = img

    start_x = int(height / 2 - clip_size[1] / 2)  # clip_size: (width, height)
    start_y = int(width / 2 - clip_size[0] / 2)
    end_x = start_x + clip_size[1]
    end_y = start_y + clip_size[0]

    rotated_img = rotated_img[start_x:end_x, start_y:end_y]  # select desired zone
    if downsample_size[0] < clip_size[0]:
        rotated_img = cv2.resize(rotated_img, downsample_size)

    return rotated_img
