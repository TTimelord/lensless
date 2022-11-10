import numpy as np
import cv2

def make_separable(Y):
    rowMeans = Y.mean(axis=1, keepdims=True)
    colMeans = Y.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    Ysep = Y - rowMeans - colMeans + allMean
    # Ysep = Y - 0.5*(rowMeans + colMeans)
    return Ysep

'''
a = np.array([[1, -1, 1, -1, -1, 1, 1, 1, -1]])
b = np.array([[1, 1, 1, -1, 1, -1, 1, -1, -1]])

mat = np.multiply(a, b.T)
mat[mat < 0] = 0
mat = make_separable(mat)
print(mat)

U, S, VT = np.linalg.svd(mat)
print(S)

mat_reconstruct = np.linalg.multi_dot([U, np.diag(S), VT])
print(mat_reconstruct)
'''

# img = cv2.imread('data/captured/test/test_processed.png')
img_1 = cv2.imread('data/captured/calibration/horizontal/3_1.png')
# img_2 = cv2.imread('data/captured/calibration/horizontal/3_2.png')
# img = img_1 - img_2


img = img[:, :, 1]
print(img.shape)
cv2.imshow("img", img)
cv2.waitKey(0)

# img = make_separable(img)
rowMeans = img.mean(axis=1, keepdims=True)
colMeans = img.mean(axis=0, keepdims=True)
allMean = rowMeans.mean()
img_sep = img - rowMeans - colMeans + allMean

U, S, VT = np.linalg.svd(img_sep)
print('sigma_0/sigma_1:', S[0]/S[1])

img_recon = U[:, 0:1] @ np.diag(S[0:1]) @ VT[0:1, :] + rowMeans + colMeans - allMean

cv2.imshow('svd_sep', img_recon.astype(np.uint8))
cv2.waitKey(0)


'''
# svd directly, proved to be incorrect
U, S, VT = np.linalg.svd(img)
print('sigma:', S)
print('sigma_0/sigma_1:', S[0]/S[1])

img_recon = S[0] * U[:, [0]] @ VT[:, [0]].T

cv2.imshow('svd', img_recon.astype(np.uint8))
cv2.waitKey(0)
'''
