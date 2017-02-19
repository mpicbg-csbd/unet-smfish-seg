"""An interactive module. Use me by copy and pasting lines into iPython!"""

# global python
from __future__ import print_function, division
from glob import glob

# global scikit
import numpy as np
import skimage.io as io
import timeit
import sklearn

import sys
sys.path.append("./models/")

# import keras_classifier as kc
import util as ut

import mnist_keras as mk
import build_featurestack as bf
import rf
import unet


# run training across a set of images

# greyscale_list = glob("./knime_test_data/data/train/grayscale/grayscale_?.tif")
greyscale_list = glob("./knime_test_data/data/train/greyscale_bg_removed/bg_removed?.tif")
greyscale_list = map(io.imread, greyscale_list)
labels_list = glob("./knime_test_data/data/train/labels/composite/vertex_labels_?.tif")
labels_list = map(io.imread, labels_list)

# ---- BUILD/IMPORT FEATURESTACKS for RANDOM FOREST TRAINING

def run_random_forest():
    gabor_list = map(bf.gabor_stack, greyscale_list)
    knime_list = glob("./knime_test_data/data/train/features/features_?.tif")
    knime_list = map(io.imread, knime_list)

    X,Y = rf.imglist_to_XY(knime_list, labels_list)
    Xtrain, Ytrain, Xtest, Ytest = ut.train_test_split(X,Y,0.2)
    rafo = rf.train_rafo_from_XY(Xtrain, Ytrain, sample_weight=sw)

    print("confusion_matrix Train:")
    Ypred_train = rafo.predict(Xtrain)
    print(sklearn.metrics.confusion_matrix(Ytrain, Ypred_train))

    print("confusion_matrix Test:")
    Ypred_test = rafo.predict(Xtest)
    print(sklearn.metrics.confusion_matrix(Ytest, Ypred_test))

# ---- Try with a UNet

def run_unet():
    # import
    X,Y,coords = unet.imglists_to_XY(greyscale_list[:-1], labels_list[:-1], step=48)

    # split and shuffle
    X_train, Y_train, X_vali, Y_vali = ut.train_test_split(X,Y,test_fraction=0.2)

    # train
    model = unet.get_unet(48,48,1)
    model = unet.trainmodel(model, X_train, Y_train, X_vali, Y_vali, nb_epoch = 5)

    import keras
    model = keras.models.load_model('./keras_model.h5')

    # predict on an image and save result
    img = greyscale_list[-1]
    label = labels_list[-1]
    X,Y,coords = unet.imglists_to_XY([img], [label], step=10)
    Y_pred = model.predict(X)
    # n,one,pixels,classes = Y_pred.shape
    Y_pred = np.argmax(Y_pred, axis=-1)
    Y_pred = Y_pred.reshape((Y_pred.shape[0], 48, 48))
    res = unet.rebuild_img_from_patches(np.zeros_like(img), Y_pred, coords)
    io.imsave('grey_res.tif', res)


# ---- Automatic CrossValidation RANDOM FORESTS

def run_crossval():

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

def run_gridsearch():
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

if __name__ == '__main__':
    run_unet()
