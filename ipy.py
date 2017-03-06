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

import label_imgs
import util
from util import sglob
import unet


# import mnist_keras as mk

# run training across a set of images

knime_train_data_greys_bgblack = lambda : sglob("data/knime_test_data/data/train/greyscale_bg_removed/bg_removed?.tif")
knime_train_data_greys = lambda : sglob("data/knime_test_data/data/train/grayscale/grayscale_?.tif")
knime_train_data_labels = lambda : sglob("data/knime_test_data/data/train/labels/composite/vertex_labels_?.tif")
knime_train_data_keras_mem_predictions = lambda : sglob("data/grayscale_?_predict.tif")
knime_predict_data_greys = lambda : sglob("data/knime_test_data/data/predict/grayscale/*.tif")

unseen_greys = lambda : sglob("data/unseen_greys/mean8/*.tif")
unseen_labels = lambda : sglob("data/unseen_labels/pooled/*.tif")
# unseen_mem_predict = lambda : sglob("data/unseen_mem_predict/*.tif")
unseen_mem_predict = lambda : sglob("data/2015*predict.tif")
unseen_seg = lambda : sglob("data/2015*seg.tif")


def imsave(fname, img, **kwargs):
    io.imsave(fname, img, compress=6, **kwargs)
    # io.imsave(fname, img, **kwargs)

def imread(fname, **kwargs):
    return io.imread(fname, plugin='tifffile', **kwargs)

# ---- HOW WE MADE THE DATA

# def make_prediction_overlays():
#     pairs = zip(unseen_seg(), unseen_labels(), unseen_greys(), unseen_seg_files())
#     def save((a,b,c, name)):
#         path, base, ext = util.path_base_ext(name)
#         new = np.stack((c,a,b), axis=0)
#         imsave(base + "_overlay" + ext, new)
#     map(save, pairs)

# def min_pool_downscale():
#     from skimage.util import view_as_windows
#     def pooled(img):
#         img = img[0]
#         print("shape: ", img.shape)
#         img = view_as_windows(img, 6, step=6)
#         return np.min(img, axis=(2,3))
#     util.apply_operation_to_imgdir("data/unseen_labels/", pooled)

def mean_downscale():
    from skimage.util import view_as_windows
    def mean8(img):
        s = img.shape
        print("shape: ", img.shape)
        if s[0] > s[1]:
            img = np.transpose(img)
        img = view_as_windows(img, 8, step=8)
        return np.mean(img, axis=(2,3)).astype(np.float32)
    util.apply_operation_to_imgdir("data/unseen_greys/", mean8)

# def rotate():
#     def rot(img):
#         s = img.shape
#         if s[0] > s[1]:
#             img = np.rot90(img, 3)
#         img -= img.min()
#         img *= 20
#         return img
#     ipy.util.apply_operation_to_imgdir("unseen_greys/", rot)


# In the end we had to rotate manually, because the different images were all
# rotated/flipped differently.

# ---- Try with a UNet

def compare_segment_predictions_with_groundtruth(segs, labels):
    "segs and labels are lists of filenames of images."
    from label_imgs import match_score_1
    def print_and_score(s_l):
        s,l = s_l
        simg = imread(s)
        limg = imread(l)
        print('\n', s)
        return match_score_1(simg, limg)
    return map(print_and_score, zip(segs, labels))

def segment_classified_images(membranes, threshold):
    "membranes is a list of filenames of membrane images."
    def get_label(img):
        img = img.astype(np.float32, copy = False)
        img = np.nan_to_num(img)
        img /= img.max()

        # threshold = threshold_otsu(img)

        # x = (1-threshold) * 0.22
        # threshold += x

        # img < threshold means the membrane takes on high values and we want the cytoplasm
        mask = np.where(img < threshold, 1, 0)
        # mask = np.where(img > threshold)

        lab_img = label(mask)[0]
        print("Number of cells: ", lab_img.max())

        # convert from int32
        lab_img = np.array(lab_img, dtype='uint16')
        return lab_img

    imgs = map(imread, membranes)
    res = map(get_label, imgs)
    for fname, img in zip(membranes, res):
        path, base, ext = util.path_base_ext(fname)
        imsave(base + '_seg' + ext, img)
        imsave(base + '_seg_preview' + ext, label_imgs.labelImg_to_rgb(img))
    return res

def train_unet(greys, labels, model=None, savedir=None):
    "greys and labels are lists of filenames of greyscale and labeled images."
    grey_imgs = [imread(x) for x in greys]
    label_imgs = [imread(x) for x in labels]

    print("Input greyscale images:")
    map(print, greys)
    print("Input label images:")
    map(print, labels)

    X,Y = unet.imglists_to_XY(grey_imgs, label_imgs)
    X,Y = unet.process_XY_for_training(X, Y)
    # this shuffles the data (if we're gonna augment data, do it now)
    train_ind, test_ind = util.subsample_ind(X, Y, test_fraction=0.2, rand_state=0)
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    X_train, Y_train, X_vali, Y_vali = X[train_ind], Y[train_ind], X[test_ind], Y[test_ind]

    # train
    if model is None:
        unet.x_width = 160
        unet.y_width = 160
        unet.step = 10
        model = unet.trainmodel(X_train, Y_train, X_vali, Y_vali, batch_size = 1, nb_epoch = 3, savedir=savedir)
    else:
        model = unet.trainmodel(X_train, Y_train, X_vali, Y_vali, model, batch_size = 1, nb_epoch = 20, savedir=savedir)
    # model.save_weights('unet_weights.h5')
    return model

def predict_unet(greys, model=None, savedir=None):
    if model is None:
        unet.x_width = 200
        unet.y_width = 200
        unet.step = 10
        model = unet.get_unet()
        model.load_weights("./unet_model_weights_checkpoint.h5")

    # for name, img in zip(unseen_greyscale_files(), unseen_greys()):
    images = map(imread, greys)
    for name, img in zip(greys, images):
        res = unet.predict_single_image(model, img, batch_size=4)
        print("There are {} nans!".format(np.count_nonzero(~np.isnan(res))))
        path, base, ext =  util.path_base_ext(name)
        if savedir is None:
            imsave(base + '_predict' + ext, res.astype(np.float32))
        else:
            imsave(savedir + "/" + base + '_predict' + ext, res.astype(np.float32))

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
