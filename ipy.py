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

import tabulate

import label_imgs
import util
import unet
import datasets
import train
import predict

import gputools    



def test_ground_truths(verbose=False, warp=False):
    """
    All the orig size ground truths have been verified!
    Hurray!
    But all the down3 cells are broken! they've lost this property :(
    """
    
    # or get it from scipy.ndimage.morphology import generate_binary_structure
    structure = [[1,1,1], [1,1,1], [1,1,1]] # this is the structure that was used by Benoit & Carine!

    files = glob('data3/labeled_data_cellseg/labels/*.tif')
    table = [["File", "# orig cells", "# cells after label", "# cells after warp"]]
    for i in range(len(files)):
        orig = io.imread(files[i])[0]
        lab = orig.copy()
        lab[lab!=0] = 1 # remove the cells
        orig_2 = label(lab, structure=structure)[0]
        lab = lab.astype('float32')
        labmax = lab.max()
        lab /= labmax
        if warp:
            deltax = np.random.normal(loc=0, scale=10, size=(4,4))
            deltay = np.random.normal(loc=0, scale=10, size=(4,4))
            lab, delta, coords = unet_warp_orig(lab, delta=(deltax, deltay))
        lab *= labmax
        l0 = lab.copy()
        l0[l0 > 0.75]=1
        l0[l0 <= 0.75]=0
        lab_new = label(l0, structure=structure)[0]
        if verbose:
            combo = np.stack((orig, lab, lab_new), axis=0)
            io.imsave('combo{:02d}.tif'.format(i), combo.astype('float32'))
        # mat = label_imgs.matching_matrix(orig, lab_new)
        # table.append([os.path.basename(files[i]), label_imgs.seg(mat), label_imgs.match_score_1(mat)])
        table.append([os.path.basename(files[i]), orig.max(), orig_2.max(), lab_new.max()])
    plt.plot([t[1] for t in table[1:]])
    plt.plot([t[2] for t in table[1:]])
    plt.plot([t[3] for t in table[1:]])
    return table

# ------------------------------------------------------------

def sum2d(img):
    sum0 = np.cumsum(img,0)
    sum1 = np.cumsum(sum0, 1)
    return sum1

def unet_warp_orig(img, delta=None):
    a,b = img.shape
    if delta:
        deltax, deltay = delta[0], delta[1]
    else:
        deltax = np.random.normal(loc=0, scale=10, size=(4,4))
        deltay = np.random.normal(loc=0, scale=10, size=(4,4))
    deltax = imresize(deltax, size=(a,b), mode='F')
    deltay = imresize(deltay, size=(a,b), mode='F')
    dxdx = np.max(np.diff(deltax, axis=0))
    dydx = np.max(np.diff(deltax, axis=1))
    dxdy = np.max(np.diff(deltay, axis=0))
    dydy = np.max(np.diff(deltay, axis=1))
    print("MAX GRADS", dxdx, dydx, dxdy, dydy)
    delta_big = np.stack((deltax, deltay), axis=0)
    coords = np.indices(img.shape)
    newcoords = delta_big + coords
    res = tform.warp(img, newcoords, order=1)
    return res, delta_big, coords

def plot_vector_field(img):
    """
    only designed to be the right scale for smooth warps of 500^2 image patches.
    VERY FRUSTRATING THAT... images have dimensions [y,x] = [vertical,horizontal]
    and the vertical axis is always plotted DOWNWARDS!!! y goes from 0 (top) to
    y_max at the bottom! This is the opposite of all other plots...
    """
    n = 10
    res, delta, coords = unet_warp_orig(img)
    plt.figure()
    plt.imshow(img[::-1])
    plt.figure()
    plt.imshow(res[::-1])
    plt.figure()
    plt.quiver(delta[1,::n,::n],
                delta[0,::n,::n],
                headlength=2,
                headwidth=2,
                headaxislength=3)
    plt.streamplot(coords[1,0,::n]/n,
                   coords[0,::n,0]/n,
                   delta[1,::n,::n],
                   delta[0,::n,::n])
    return "Awesome plot, dude!"

def warp_gaussian(img, stdev=2, w=10):
    """
    warp img according to random gaussian vector field
    w is window width.
    using w/stdev ~= 5 gives MAX GRAD approx 1.0 (and this continues linearly)
    in the unet paper they use large patches (how large?)... and a smooth warp
    field upscaled from 3x3 (roughly a factor of 100?) maybe more?... giving a 
    ratio of 100/10 ~= 10 and a max gradient of roughly 0.5... nice and smooth :)
    """
    # img[np.isnan(img)] = 0
    # assert type(w)==int
    a,b = img.shape
    da,ra = divmod(a, w)
    db,rb = divmod(b, w)
    
    deltax = np.random.normal(loc=0, scale=stdev, size=(da,db))
    deltay = np.random.normal(loc=0, scale=stdev, size=(da,db))
    deltax = imresize(deltax, size=(a,b), mode='F')
    deltay = imresize(deltay, size=(a,b), mode='F')
    dxdx = np.max(np.diff(deltax, axis=0))
    dydx = np.max(np.diff(deltax, axis=1))
    dxdy = np.max(np.diff(deltay, axis=0))
    dydy = np.max(np.diff(deltay, axis=1))
    # print("MAX GRADS", dxdx, dydx, dxdy, dydy)

    delta2 = np.stack((deltax, deltay), axis=0)
    coords = np.indices(img.shape)
    newcoords = delta2 + coords
    res = tform.warp(img, newcoords, order=3)
    return res

def random_augmentation(patch):
    """
    flip, rotate, and warp with some probability
    """
    if random.random() < 0.5:
        patch = np.flip(patch, axis=1) # axis=1 is the horizontal axis
    randangle = (random.random()-0.5)*60 # even dist between ± 30
    rotate(patch, randangle)
    if random.random() < 0.9:
        patch = warp_gaussian(patch, stdev=2, w=10) # good for full-resolution images!
    return patch

def explore_warps(img):
    # ss = np.linspace(0, 2, 8)
    ss = [2]*30
    ws = [10]
    big = np.zeros((len(ss), len(ws)) + img.shape)

    x,y = 0,0
    for s in ss:
        y=0
        for w in ws:
            # res = warp_gaussian(img, stdev=s, w=w)
            res = unet_warp_orig(img)
            big[x, y] = res
            y+=1
        x+=1
    return big

# io.imsave('big.tif', big.astype('float32'), metadata={'axes':'CTYX'}, imagej=True)

def explore_warps_multisize(img):
    """
    when you want to make a 2d tiling of images with different (but similar) sizes.
    """
    x,y = 0,0
    a,b = img.shape
    big = np.zeros(shape=(a*7, b*11))
    print(big.shape)
    # assert 0
    for s in np.linspace(0, 3, 4):
        y=0
        for w in range(5,15,3):    
            res, sumy, sumx = warp(img, mean=0, stdev=s, w=w)
            a2,b2 = res.shape
            print(a2,b2)
            print()
            big[x:x+a2, y:y+b2] = res
            y+=b
        x+=a
    return big

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
