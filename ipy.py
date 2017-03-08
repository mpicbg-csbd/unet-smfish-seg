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

# --- NEW DATA BELOW [only use this] ---

greyscales = lambda : sglob("data2/greyscales/*.tif")
labels = lambda : sglob("data2/Cell_segmentations_paper/*.tif")
# newgreys = lambda : sglob("data2/20150513_New_data/*.tif")
# greyscales_down3x = lambda : sglob("data2/labeled_data_100xObj/images/down3x/*.tif")
# labels_down3x = lambda : sglob("data2/labeled_data_100xObj/labels/pooled/*.tif")



def imsave(fname, img, **kwargs):
    io.imsave(fname, img, compress=6, plugin='tifffile', **kwargs)
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

def min_pool_downscale():
    from skimage.util import view_as_windows
    def pooled(img):
        img[img==0] = 3 # so that membrane is 1 is the min value
        print("shape: ", img.shape)
        img = view_as_windows(img, 3, step=3)
        img = np.min(img, axis=(2,3))
        # now permute back
        img[img==3] = 0
        return img
    util.apply_operation_to_imgdir("data2/labeled_data_100xObj/labels/", pooled)

def mean_downscale():
    from skimage.util import view_as_windows
    def down6x(img):
        s = img.shape
        print("shape: ", img.shape)
        # if s[0] > s[1]:
        # img = np.transpose(img)
        img = view_as_windows(img, 6, step=6)
        img = np.mean(img, axis=(2,3)).astype(np.float32)
        return img/img.max()
    util.apply_operation_to_imgdir("data2/labeled_data_100xObj/originals/", down6x)

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

def upscale_and_compare(labeling, annotated):
    a,b = labeling.shape
    _,c,d = annotated.shape
    upscaled = zoom(labeling, (c/a, d/b), order=0)
    score = label_imgs.match_score_1(annotated[0], upscaled)
    imsave('upscaled.tif', upscaled)
    imsave('cells.tif', annotated[0])
    return score

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

def get_label(img, threshold):
    "normalizes img min&max to [0,1), then binarize at threshold, then labels connected components."
    img = img.astype(np.float32, copy = False)
    img = np.nan_to_num(img) # sets nan to zero?
    img /= img.max()

    # threshold = threshold_otsu(img)

    # x = (1-threshold) * 0.22
    # threshold += x

    # img < threshold means the membrane takes on high values and we want the cytoplasm
    mask = np.where(img < threshold, 1, 0)

    lab_img = label(mask)[0]
    print("Number of cells: ", lab_img.max())

    # convert from int32
    lab_img = np.array(lab_img, dtype='uint16')
    return lab_img

def segment_classified_images(membranes, threshold):
    "membranes is a list of filenames of membrane images."
    imgs = [imread(mem) for mem in membranes]
    res = [get_label(img, threshold) for img in imgs]
    for fname, img in zip(membranes, res):
        path, base, ext = util.path_base_ext(fname)
        imsave(base + '_seg' + ext, img) 
        imsave(base + '_seg_preview' + ext, label_imgs.labelImg_to_rgb(img))
    return res

def train_unet(greys, labels, model, savedir=None):
    "greys and labels are lists of filenames of greyscale and labeled images."
    # We're training on only the right half of each image!
    # Then we can easily identify overfitting by eye.

    grey_imgs = [imread(x) for x in greys]
    label_imgs = [imread(x)[0] for x in labels]
    grey_imgs_small = []
    label_imgs_small = []
    for grey,lab in zip(grey_imgs, label_imgs):
        a,b = grey.shape    
        grey_imgs_small.append(grey[:,0:b//2])
        label_imgs_small.append(lab[:,0:b//2])

    print("Input greyscale images:")
    for name in greys:
        print(name)
    print("Input label images:")
    for name in labels:
        print(name)

    for grey,lab in zip(grey_imgs, label_imgs):
        print(grey.shape, lab.shape)
    for grey,lab in zip(grey_imgs_small, label_imgs_small):
        print(grey.shape, lab.shape)

    X,Y = unet.imglists_to_XY(grey_imgs, label_imgs)
    X,Y = unet.process_XY_for_training(X, Y)
    # this shuffles the data (if we're gonna augment data, do it now)
    print("X.shape = ", X.shape, " and Y.shape = ", Y.shape)
    train_ind, test_ind = util.subsample_ind(X, Y, test_fraction=0.2, rand_state=0)
    print("train_ind = ", train_ind, " and test_ind =", test_ind)
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    X_train, Y_train, X_vali, Y_vali = X[train_ind], Y[train_ind], X[test_ind], Y[test_ind]

    model = unet.trainmodel(X_train, Y_train, X_vali, Y_vali, model, batch_size = 1, nb_epoch = 300, savedir=savedir)
    return model

def predict_unet(greys, model, savedir=None):
    # for name, img in zip(unseen_greyscale_files(), unseen_greys()):
    images = map(imread, greys)
    for name, img in zip(greys, images):
        res = unet.predict_single_image(model, img, batch_size=4)
        # print("There are {} nans!".format(np.count_nonzero(~np.isnan(res))))
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
