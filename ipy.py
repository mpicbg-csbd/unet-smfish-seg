"""An interactive module. Use me by copy and pasting lines into iPython!"""

# global python
from __future__ import print_function, division
from glob import glob

# global scikit
import numpy as np
import skimage.io as io
import timeit

# import keras_classifier as kc
# import mnist_keras as mk
import rf
import build_featurestack as bf
import sklearn


# run training across a set of images

# greyscale_list = glob("./knime_test_data/data/train/grayscale/grayscale_?.tif")
greyscale_list = glob("./knime_test_data/data/train/greyscale_bg_removed/bg_removed?.tif")
greyscale_list = map(io.imread, greyscale_list)
labels_list = glob("./knime_test_data/data/train/labels/composite/vertex_labels_?.tif")
labels_list = map(io.imread, labels_list)
gabor_list = map(bf.gabor_stack, greyscale_list)
knime_list = glob("./knime_test_data/data/train/features/features_?.tif")
knime_list = map(io.imread, knime_list)

X,Y = rf.imglist_to_XY(knime_list, labels_list)

# samples = 7 * Y.shape[0] // 8
# subsample = np.random.choice(Y.shape[0], size=samples, replace=False)
# Xss,Yss = X[subsample,:], Y[subsample]

# preprocess data
# potentially remove zeros?
# shuffle
# mask = (X[:,0]!=0) * (Y==0)
Xm, Ym = X, Y
# Xm,Ym = X[mask,:], Y[mask]
# multidimensional arrays are only permuted along first axis (in place)
perm = np.random.permutation(Xm.shape[0])
Xm, Ym = Xm[perm, :], Ym[perm]

s = Ym.shape[0]
ind = 6 * s // 7
(Xtrain, Ytrain), (Xtest, Ytest) = (Xm[:ind, :], Ym[:ind]), (Xm[ind:, :], Ym[ind:])

rafo = rf.train_rafo_from_XY(Xtrain, Ytrain, sample_weight=sw)

print("confusion_matrix Train:")
Ypred_train = rafo.predict(Xtrain)
print(sklearn.metrics.confusion_matrix(Ytrain, Ypred_train))

print("confusion_matrix Test:")
Ypred_test = rafo.predict(Xtest)
print(sklearn.metrics.confusion_matrix(Ytest, Ypred_test))

# ---- Try with a UNet

import unet

mod = unet.get_unet(1,48,48)

# ---- Automatic CrossValidation

import xgboost as xgb
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error

def crossval(X,Y,n_folds=3):
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

def grid_search():
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
