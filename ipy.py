"""An interactive module. Use me by copy and pasting lines into iPython!"""

# import sklearn
# from skimage.filter import threshold_isodata, threshold_otsu

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
plt.ion()
from scipy.ndimage import label, zoom, rotate
from scipy.misc import imresize
import skimage.transform as tform

import timeit
import time
from glob import glob
import sys
import os
import re
import math
import random

from tabulate import tabulate

import segtools as st
import util
import unet
import datasets
import train
import predict
import warping
import gputools


structure = [[1,1,1], [1,1,1], [1,1,1]] # this is the structure that was used by Benoit & Carine!

def play_with_matchings():
    lab1 = io.imread('data3/labeled_data_cellseg/labels/20150430_eif4g_dome01_R3D_MASKS.tif')[0]
    lab2 = warping.warp_label_img(lab1)
    mat = st.pixel_sharing_graph(lab1, lab2)
    mo = st.matching_overlap(mat)
    perm = st.permutation_from_matching(mo)
    permed = st.permute_img(img, perm)
    res = np.stack((lab1, lab2, permed))
    return res

def test_ground_truths(verbose=False, warp_scale=0):
    """
    All the orig size ground truths have been verified!
    Hurray!
    But all the down3 cells are broken! they've lost this property :(
    And the down6 as well obviously.
    """

    files = glob('data3/labeled_data_cellseg/labels/*.tif')
    table = [["File", "# orig cells", "# cells after label", "# cells after warp"]]
    for i in range(len(files[:4])):
        orig = io.imread(files[i])[0]
        warpable = orig.copy()
        warpable[warpable!=0] = 1 # remove the cells
        relabeled = label(warpable, structure=structure)[0]
        warped_relabeled = warping.warp_label_img(orig, warp_scale=20, w=4)
        if verbose:
            combo = np.stack((orig, relabeled, warped_relabeled), axis=0)
            io.imsave('combo{:02d}.tif'.format(i), combo.astype('float32'))
        # mat = st.img_matching_pixelwise(orig, warped_relabeled)
        # table.append([os.path.basename(files[i]), st.seg(mat), st.match_score_1(mat)])
        table.append([os.path.basename(files[i]), orig.max(), relabeled.max(), warped_relabeled.max()])

    plt.plot([t[1] for t in table[1:]], label="orig")
    plt.plot([t[2] for t in table[1:]], label="relabeled")
    plt.plot([t[3] for t in table[1:]], label="warped")
    names = [t[0][-22:-10] for t in table[1:]]
    xticks, xlabels = plt.xticks()
    plt.xticks(range(len(names)), names, rotation=70)
    plt.legend()
    return table

# ------------------------------------------------------------

def sum2d(img):
    sum0 = np.cumsum(img,0)
    sum1 = np.cumsum(sum0, 1)
    return sum1

# ------------------------------------------------------------

def save_patches(X,Y,Ypred):
    # import matplotlib.pyplot as plt
    # def imsho(x, fname):
    #     # plt.figure()
    #     # plt.imshow(x, interpolation='nearest')
    #     imsave(fname, x)
    idx = np.random.randint(Ypred.shape[0])
    x= X[idx,0]
    a,b = x.shape
    # imsho(x, 'x.tif')
    y_gt = Y[idx,:,0].reshape((a,b))
    # imsho(y_gt)
    y_pre = Ypred[idx,:,0].reshape((a,b))
    # imsho(y_pre)
    imsave('randstack.tif', np.stack((x,y_gt,y_pre), axis=0))

# ---- BUILD/IMPORT FEATURESTACKS for RANDOM FOREST TRAINING

def train_and_test_rafo_gabor(greys, labels):
    import build_featurestack as bf
    import rf
    greyscale_list = map(imread, greys)
    labels_list = map(imread, labels)

    gabor_list = map(bf.gabor_stack, greyscale_list)

    X,Y = rf.imglist_to_XY(gabor_list, labels_list)
    train_ind, test_ind = util.subsample_ind(X,Y,0.2)
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    X_train, Y_train, X_vali, Y_vali = X[train_ind], Y[train_ind], X[test_ind], Y[test_ind]
    sw = rf.balanced_sample_weights(Ytrain)
    rafo = rf.train_rafo_from_XY(Xtrain, Ytrain, sample_weight=sw)

    print("confusion_matrix Train:")
    Ypred_train = rafo.predict(Xtrain)
    print(sklearn.metrics.confusion_matrix(Ytrain, Ypred_train))

    print("confusion_matrix Test:")
    Ypred_test = rafo.predict(Xtest)
    print(sklearn.metrics.confusion_matrix(Ytest, Ypred_test))

def train_and_test_rafo_weka():
    import rf

    knime_list = util.sglob("data/knime_test_data/data/train/features/features_?.tif")
    knime_list = map(imread, knime_list)

    X,Y = rf.imglist_to_XY(knime_list, labels_list)
    train_ind, test_ind = util.subsample_ind(X,Y,0.2)
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    X_train, Y_train, X_vali, Y_vali = X[train_ind], Y[train_ind], X[test_ind], Y[test_ind]

    sw = rf.balanced_sample_weights(Y_train)
    rafo = rf.train_rafo_from_XY(X_train, Y_train, sample_weight=sw)

    print("confusion_matrix Train:")
    Ypred_train = rafo.predict(X_train)
    print(sklearn.metrics.confusion_matrix(Y_train, Ypred_train))

    print("confusion_matrix Test:")
    Ypred_test = rafo.predict(X_test)
    print(sklearn.metrics.confusion_matrix(Y_test, Ypred_test))

# ---- Automatic CrossValidation RANDOM FORESTS

def run_crossval(X, Y, n_folds=3):

    import xgboost as xgb
    from sklearn.cross_validation import KFold, train_test_split
    from sklearn.metrics import confusion_matrix, mean_squared_error

    rng = np.random.RandomState(31337)
    kf = KFold(Y.shape[0], n_folds=n_folds, shuffle=True, random_state=rng)
    for train_index, test_index in kf:
        sw = rf.balanced_sample_weights(Y[train_index])
        xgb_model = xgb.XGBClassifier().fit(X[train_index],Y[train_index], sample_weight=sw)

        print("confusion_matrix Train:")
        Ypred_train = xgb_model.predict(X[train_index])
        print(sklearn.metrics.confusion_matrix(Y[train_index], Ypred_train))

        print("confusion_matrix Test:")
        Ypred_test = xgb_model.predict(X[test_index])
        print(sklearn.metrics.confusion_matrix(Y[test_index], Ypred_test))

        print("--------------------")

# ---- Grid Search through params

def run_gridsearch():
    param_test1 = {
     'max_depth':range(3,10,2),
     'min_child_weight':range(1,6,2)
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140,
                            max_depth=5,
                             min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                             objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                             param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(train[predictors],train[target])
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
