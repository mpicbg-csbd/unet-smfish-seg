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

import sys
sys.path.append("./models/")

import util

# import mnist_keras as mk

# run training across a set of images

greyscale_list_files = glob("./knime_test_data/data/train/grayscale/grayscale_?.tif")
# greyscale_list_files = glob("./knime_test_data/data/train/greyscale_bg_removed/bg_removed?.tif")
greyscale_list = map(io.imread, greyscale_list_files)
labels_list_files = glob("./knime_test_data/data/train/labels/composite/vertex_labels_?.tif")
labels_list = map(io.imread, labels_list_files)

# ---- BUILD/IMPORT FEATURESTACKS for RANDOM FOREST TRAINING

def train_and_test_rafo_gabor():
    import build_featurestack as bf
    import rf

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

    knime_list = glob("./knime_test_data/data/train/features/features_?.tif")
    knime_list = map(io.imread, knime_list)

    X,Y = rf.imglist_to_XY(knime_list, labels_list)
    Xtrain, Ytrain, Xtest, Ytest = util.train_test_split(X,Y,0.2)
    sw = rf.balanced_sample_weights(Ytrain)
    rafo = rf.train_rafo_from_XY(Xtrain, Ytrain, sample_weight=sw)

    print("confusion_matrix Train:")
    Ypred_train = rafo.predict(Xtrain)
    print(sklearn.metrics.confusion_matrix(Ytrain, Ypred_train))

    print("confusion_matrix Test:")
    Ypred_test = rafo.predict(Xtest)
    print(sklearn.metrics.confusion_matrix(Ytest, Ypred_test))

# ---- Try with a UNet

def train_unet(model=None):
    import unet
    unet.x_width = 48
    unet.y_width = 48
    unet.step = 24
    X,Y = unet.imglists_to_XY(greyscale_list[:-1], labels_list[:-1])
    # train
    if model is None:
        model = unet.trainmodel(X, Y, batch_size = 32, nb_epoch = 100)
    else:
        model = unet.trainmodel(X, Y, model, batch_size = 32, nb_epoch = 5)
    # model.save_weights('unet_weights.h5')
    return model

def predict_unet(model=None):
    import unet
    unet.x_width = 48
    unet.y_width = 48
    unet.step = 6

    if model is None:
        model = unet.get_unet()
        model.load_weights('unet_model_weights_checkpoint.h5')

    for name, img in zip(greyscale_list_files, greyscale_list):
        res = unet.predict_single_image(model, img)
        path, base, ext =  util.path_base_ext(name)
        io.imsave(base + '_predict' + ext, res)

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
    train_and_test_rafo_weka()
