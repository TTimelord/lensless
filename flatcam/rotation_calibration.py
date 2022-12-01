from phi_get_new import *
from flatcam import make_separable

""" 
Rotation test 1 (should be done every time after the camera is removed from the bracket): 
rotate the image obtained about axis perpendicular to the plane to align it horizontally
(that is to find the rotation angle resulting in the largest singular value ratio, 
which indicate image's good separability)
"""

img = cv2.imread('data/captured/test/test_origin.png')
# img = cv2.imread('data/captured/calibration/horizontal/9_2.png')
# img_2 = cv2.imread('data/captured/calibration/horizontal/100_1.png')

print(img.shape)
# matrix_gr[matrix_gr > 0.3] = 0.9
# matrix_gr[matrix_gr < 0.3] = 0.01
img = img[:, :, 1]
print(img.shape)
cv2.imshow("img", img)
cv2.waitKey(0)

img = make_separable(img)

height, width = img.shape

""" Rotate the image at small interval and find the angle corresponding to the best singular value ratio """
best_ratio, best_angle = 1, -100
for i in range(100):
    angle = i * 0.01 - 1
    rotate_mat = cv2.getRotationMatrix2D((int(width/2), int(height/2)), angle, 1)  # center pos, rotation angle, zoom ratio
    rotated_img = cv2.warpAffine(img, rotate_mat, (int(width), int(height)))  # image, rotation matrix, image size after rotation
    rotated_img = rotated_img[int(height/10):-int(height/10), (int(width/10)):-int(width/10)]  # Cut the edge (10:-10) means cutting off upper and lower 10 pixels)
    # cv2.imshow("rotated", rotated_img)
    # cv2.waitKey(0)

    Y = make_separable(rotated_img)
    U, sigma, VT = np.linalg.svd(Y)

    # Record best singular value ratio and corresponding angle
    if sigma[0] / sigma[1] > best_ratio:
        best_ratio = sigma[0] / sigma[1]
        best_angle = angle
    print(f"angle: {angle}, ratio: {sigma[0] / sigma[1]}")

print(f"best angle: {best_angle}, best ratio: {best_ratio}")

rotate_mat = cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), best_angle,
                                     1)  # center pos, rotation angle, zoom ratio
rotated_img = cv2.warpAffine(img, rotate_mat,
                             (int(width), int(height)))  # image, rotation matrix, image size after rotation
rotated_img = rotated_img[int(height / 10):-int(height / 10),
              (int(width / 10)):-int(width / 10)]  # Cut the edge (10:-10 means cutting off upper and lower 10 pixels)
cv2.imshow("rotated", rotated_img)
cv2.waitKey(0)


Y = make_separable(img)
U = SVD_getu(Y)


""" Rotate test 2 """
# angle_max = 1
# angle = angle_maxCalibration images/Horizontal/20_2
# num = 50
# best_ratio, best_angle = 1, -100
# for i in range(1, num+1):
#     name = "Rotate/" + str(i) + ".png"
#     # matrix = cv2.imread(name)[:, :, 0]
#     matrix = mpimg.imread(name)
#     matrix_gr = matrix[1::2, 0::2]
#
#     rotate_mat = cv2.getRotationMatrix2D((160, 120), -0.41, 1)  # center pos, rotation angle, zoom ratio
#     rotated_img = cv2.warpAffine(matrix_gr, rotate_mat, (320, 240))
#     rotated_img = rotated_img[10:-10, 20:-20]
#
#     Y = make_separable(rotated_img)
#     U, sigma, VT = np.linalg.svd(Y)
#     if sigma[0] / sigma[1] > best_ratio:
#         best_ratio = sigma[0] / sigma[1]
#         best_angle = angle
#     print(f"angle: {angle}, ratio: {sigma[0] / sigma[1]}")
#     angle -= 2 * angle_max / num
#
# print(f"best angle: {best_angle}, best ratio: {best_ratio}")
