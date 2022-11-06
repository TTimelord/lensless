from image_generation import *

""" Generate image for calibration """
def cali_gen(N):
    for i in range(1, N+1):
        in_pic_1 = "data/display/calibration/horizontal/" + str(i) + "_h_1.png"
        out_pic_1 = "data/display/calibration/horizontal/" + str(i) + "_1.png"
        in_pic_2 = "data/display/calibration/horizontal/" + str(i) + "_h_2.png"
        out_pic_2 = "data/display/calibration/horizontal/" + str(i) + "_2.png"
        image_generation(in_pic_1, out_pic_1, (550, 550))
        image_generation(in_pic_2, out_pic_2, (550, 550))
        print(out_pic_1 + " and " + out_pic_2 + " is saved")

    for i in range(1, N+1):
        in_pic_1 = "data/display/calibration/vertical/" + str(i) + "_v_1.png"
        out_pic_1 = "data/display/calibration/vertical/" + str(i) + "_1.png"
        in_pic_2 = "data/display/calibration/vertical/" + str(i) + "_v_2.png"
        out_pic_2 = "data/display/calibration/vertical/" + str(i) + "_2.png"
        image_generation(in_pic_1, out_pic_1, (550, 550))
        image_generation(in_pic_2, out_pic_2, (550, 550))
        print(out_pic_1 + " and " + out_pic_2 + " is saved")

    print("Process finished!")

""" Generate image for collecting data-set """

# data = open("load.txt")
# lines = data.readlines()
# for line in lines:
#     line = line.strip('\n')
#     in_pic, out_pic = line, line
#     image_generation(in_pic, out_pic)
#     print(out_pic + " is saved")

""" Generate image for the test of sensor's field of vision """
# green = np.zeros((100, 100, 3))
# green[:, :, 1] = 255*np.ones((100, 100))
# cv2.imwrite("green.png", green)
#
# image_generation("green.png", "green.png", (20, 20))

# l = 0
# i = 1
# while l <= 700:
#     name = "green/" + str(i) + ".png"
#     image_generation_2("green.png", name, (100, 100), l)
#     l += 20
#     i += 1
#     print(name + " is saved")

""" Generate rotated image """
# img = cv2.imread("Calibration images/Horizontal/21_h.png")
# width, height = img.shape[1], img.shape[0]
# angle_max = 1
# angle = angle_max
# num = 50
# for i in range(1, num+1):
#     rotate_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)  # center pos, rotation angle, zoom ratio
#     rotated_img = cv2.warpAffine(img, rotate_mat, (width, height))
#     rotated_img = rotated_img[3:-3, 3:-3]
#     # cv2.imshow("rotated", rotated_img)
#     # cv2.waitKey(0)
#     name = "Calibration images/Rotate/" + str(i) + ".png"
#     image_generation_3(rotated_img, name)
#     angle -= 2*angle_max/num
#     print(angle)

""" Generate display image for numbers """
# for i in range(10):
#     name_in = "num/" + str(i) + "_2.png"
#
#     """ Cut """
#     # matrix = cv2.imread(name_in)
#     # matrix = matrix[2:-2, 5:-5]
#     # cv2.imwrite(name_in, matrix)
#
#     name_out = "num/" + str(i) + "_d.png"
#     image_generation(name_in, name_out, (550, 550))
#     print(name_out + " is generated！")

""" Generate display image for a single image """
# image_generation("cross.png", "cross.png", (550, 550))

""" Test images """
# for i in range(1, 19):
#     name_in = "test images/" + str(i) + ".png"
#     name_out = "test images/" + str(i) + "_d.png"
#     image_generation(name_in, name_out, (550, 550))
    




