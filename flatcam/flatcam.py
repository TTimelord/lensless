import numpy as np
import math
from scipy.ndimage import rotate as imrotate
from numpy.linalg import multi_dot


def clean_calib( calib ):
    # Fix any formatting issues from Matlab to Python
    calib['cSize'] = np.squeeze(calib['cSize'])
    calib['angle'] = np.squeeze(calib['angle'])


# 装载标定矩阵
def obtain_calib_svd( calib ):
    calib_svd = calib # note, what happens to calib_svd happens to calib. To make just a copy, calib_svd = dict(calib)
    clean_calib(calib_svd)
    P1 = np.dstack((calib['P1b'], calib['P1g'], calib['P1r']))  #横条纹拆分成3通道，分别标定出矩阵phiL
    Q1 = np.dstack((calib['Q1b'], calib['Q1g'], calib['Q1r']))  #竖条纹拆分成3通道，分别标定出矩阵phiR
    # Initialize new entries for calib data struct
    calib_svd['UL_all'] = np.empty([P1.shape[0], P1.shape[0], 3])
    calib_svd['DL_all'] = np.empty([P1.shape[0], P1.shape[1], 3])
    calib_svd['VL_all'] = np.empty([P1.shape[1], P1.shape[1], 3])
    calib_svd['singL_all'] = np.empty([P1.shape[1], 3])
    calib_svd['UR_all'] = np.empty([Q1.shape[0], Q1.shape[0], 3])
    calib_svd['DR_all'] = np.empty([Q1.shape[0], Q1.shape[1], 3])
    calib_svd['VR_all'] = np.empty([Q1.shape[1], Q1.shape[1], 3])
    calib_svd['singR_all'] = np.empty([Q1.shape[1], 3])
    for i in range(3):  # 将6个标定矩阵赋值到空矩阵中
        # Left matrices (P1)
        u, s, vh = np.linalg.svd(P1[:, :, i], full_matrices=True)
        calib_svd['UL_all'][:, :, i] = u
        calib_svd['DL_all'][:, :, i] = np.concatenate((np.diag(s), np.zeros([P1.shape[0] - s.size, s.size])))
        calib_svd['VL_all'][:, :, i] = vh.T
        calib_svd['singL_all'][:, i] = s
        # Right matrices (Q1)
        u, s, vh = np.linalg.svd(Q1[:, :, i], full_matrices=True)
        calib_svd['UR_all'][:, :, i] = u
        calib_svd['DR_all'][:, :, i] = np.concatenate((np.diag(s), np.zeros([Q1.shape[0] - s.size, s.size])))
        calib_svd['VR_all'][:, :, i] = vh.T
        calib_svd['singR_all'][:, i] = s
    print(calib_svd['UL_all'])


# 中心化矩阵
def make_separable(Y):
    rowMeans = Y.mean(axis=1, keepdims=True)
    colMeans = Y.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    Ysep = Y - rowMeans - colMeans + allMean
    return Ysep


def fcrecon(cap, calib, lmbd):
    # check if SVDs have been taken
    if not 'UL_all' in calib:
        obtain_calib_svd(calib)
    Y = fc2bayer(cap, calib)  # convert RAW output to Bayer color channels
    Y = make_separable(Y) # let rows and columns have 0-mean
    X_bayer = np.empty([calib['VL_all'].shape[0], calib['VR_all'].shape[0], 4])
    for c in range(4):
        UL = calib['UL_all'][:, :, c]
        DL = calib['DL_all'][:, :, c]
        VL = calib['VL_all'][:, :, c]
        singLsq = np.square(calib['singL_all'][:, c])
        UR = calib['UR_all'][:, :, c]
        DR = calib['DR_all'][:, :, c]
        VR = calib['VR_all'][:, :, c]
        singRsq = np.square(calib['singR_all'][:, c])
        Yc = Y[:, :, c]
        inner = multi_dot([DL.T,UL.T,Yc,UR,DR]) / (np.outer(singLsq, singRsq) + np.full(X_bayer.shape[0:1], lmbd))
        X_bayer[:, :, c] = multi_dot([VL, inner, VR.T])
    print(X_bayer.max(), X_bayer.min())
    X_bayer = X_bayer.clip(min=0)  # non-negative constraint: set all negative values to 0   -0.4 2.6
    return bayer2rgb(X_bayer, True)  # bring back to RGB and normalize


def fcrecon_new(img_cap, calib, lmbd):
    # check if SVDs have been taken
    if 'UL_all' not in calib:
        obtain_calib_svd(calib)
    for i in range(3):
        img_cap[:, :, i] = make_separable(img_cap[:, :, i])

    X = np.zeros([calib['VL_all'].shape[0], calib['VR_all'].shape[0], 3])
    for c in range(3):
        UL = calib['UL_all'][:, :, c]
        DL = calib['DL_all'][:, :, c]
        VL = calib['VL_all'][:, :, c]
        singLsq = np.square(calib['singL_all'][:, c])
        UR = calib['UR_all'][:, :, c]
        DR = calib['DR_all'][:, :, c]
        VR = calib['VR_all'][:, :, c]
        singRsq = np.square(calib['singR_all'][:, c])
        Yc = img_cap[:, :, c]
        inner = multi_dot([DL.T, UL.T, Yc, UR, DR]) / (np.outer(singLsq, singRsq) + np.full(X.shape[0:1], lmbd))
        X[:, :, c] = multi_dot([VL, inner, VR.T])

    # X = X.clip(min=0)
    return X