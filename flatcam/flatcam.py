import numpy as np
from numpy.linalg import multi_dot
from utils import process_img

def clean_calib( calib ):
    # Fix any formatting issues from Matlab to Python
    calib['clipSize'] = np.squeeze(calib['clipSize'])
    calib['downsampleSize'] = np.squeeze(calib['downsampleSize'])
    calib['angle'] = np.squeeze(calib['angle'])


# 装载标定矩阵
def obtain_calib_svd(calib):
    calib_svd = calib # note, what happens to calib_svd happens to calib. To make just a copy, calib_svd = dict(calib)
    clean_calib(calib_svd)
    P1 = np.dstack((calib['P1b'], calib['P1g'], calib['P1r']))  # 横条纹拆分成3通道，分别标定出矩阵phiL
    Q1 = np.dstack((calib['Q1b'], calib['Q1g'], calib['Q1r']))  # 竖条纹拆分成3通道，分别标定出矩阵phiR
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


def make_separable(Y):
    """
    By subtracting row and col mean, convert sensor response back to separable image (rank-1)
    Args:
        Y: original img

    Returns:
        Ysep: separable img
    """
    rowMeans = Y.mean(axis=1, keepdims=True)
    colMeans = Y.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    Ysep = Y - rowMeans - colMeans + allMean
    return Ysep


def fcrecon(img_cap, calib, lmbd):
    # check if SVDs have been taken
    if 'UL_all' not in calib:
        obtain_calib_svd(calib)
    Y = process_img(img_cap, calib['angle'], calib['clipSize'], calib['downsampleSize'])
    for i in range(3):
        Y[:, :, i] = make_separable(Y[:, :, i])

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
        Yc = Y[:, :, c]
        inner = multi_dot([DL.T, UL.T, Yc, UR, DR]) / (np.outer(singLsq, singRsq) + np.full(X.shape[0:1], lmbd))
        X[:, :, c] = multi_dot([VL, inner, VR.T])

    # X = X.clip(min=0)
    return X