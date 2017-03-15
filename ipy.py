"""An interactive module. Use me by copy and pasting lines into iPython!"""

# for old ipython
# %load_ext autoreload
# %autoreload 2

# global python
from __future__ import print_function, division
from glob import glob

# global scikit
import numpy as np
import skimage.io as io
import timeit
import sklearn
from scipy.ndimage import label, zoom
from skimage.filter import threshold_isodata, threshold_otsu

import sys
sys.path.append("./models/")
import os

import label_imgs
import util
from util import sglob
import unet




# ---- Try with a UNet

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

    knime_list = sglob("data/knime_test_data/data/train/features/features_?.tif")
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
