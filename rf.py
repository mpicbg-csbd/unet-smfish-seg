from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from glob import glob

import skimage.io as io

import os

from sklearn.ensemble import RandomForestClassifier

# FIXME: These lists should be combined.

dont_use_list = ['vsobel', 'hsobel', 'vscharr', 'vprewitt', 'hprewitt', 'roberts_positive_diagonal', 'roberts_negative_diagonal',
    'hscharr', 'deprecated', 'copy_func', 'LPIFilter2D', 'gabor_filter', 'gabor_kernel', 'gabor']

dontusenames = ['median',
 'gaussian',
 'wiener',
 'threshold_yen',
 'threshold_adaptive',
 'rank_order',
 'inverse',
 'gaussian_filter',
 'threshold_otsu',
 'gabor',
 'threshold_li',
 'threshold_isodata']

notgoodnames = dont_use_list + dontusenames

def path_base_ext(fname):
    directory, base = os.path.split(fname)
    base, ext = os.path.splitext(base)
    return directory, base, ext

def safe_makedirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def apply_operation_to_imgdir(imgdir, func, dtype='input'):
    for file in glob(imgdir + "/*.tif"):
        img = io.imread(file)
        dir, base, ext = path_base_ext(file)
        result_img = func(img)
        # result_img = np.concatenate((img, result_img[:,:, np.newaxis]), axis=2)
        newpath = dir + os.sep + func.__name__ + os.sep
        safe_makedirs(newpath)
        new_img_name = newpath + base + ext
        print("Saving to: ", new_img_name)
        if dtype == 'input':
            io.imsave(new_img_name, result_img)
        else:
            io.imsave(new_img_name, result_img.astype(dtype))

def save_all_scikit_filters():
    # [io.imsave(name, f(img)) for f in dir(F)]
    import skimage.filters as F
    import skimage.feature.blob as B

    seg_methods = [B.blob_doh, B.blob_log, B.blob_dog, B.arccos, B.gaussian_filter, B.gaussian_laplace, B.hypot, B.integral_image]


    fname = "../knime_test_data/data/train/grayscale/grayscale_0.tif"
    dir, base, ext = path_base_ext(fname)
    img = io.imread(fname)
    img /= img.max()

    # I want to make a feature stack. But I run into errors. To get around this I need
    # to manually select which filters I use and how I call them...

    # If you don't want to import the module...
    from inspect import getmembers, isfunction
    functions_list = [o for o in getmembers(F) if isfunction(o[1]) if o[1] != 'copy_func']

    funclist = [(name, f) for (name, f) in F.__dict__.iteritems() if callable(f) if name not in notgoodnames]

    for name, f in funclist:
        new_name = "../resources/" + base + '_' + name + ext
        try:
            io.imsave(new_name, f(img).astype('float32'))
        except (TypeError, ValueError, AttributeError, AssertionError) as e:
            print(); print();
            print(new_name)
            dontusenames.append(name)
            print(type(e), e)

class State:
    def __init__(self, home):
        self.home = home
        # self.home = "../knime_test_data/data/train/"
        self.grayscale_imgs = glob(home + "/grayscale/*.tif")
        self.label_imgs = glob(home + "/labels/composite/*.tif")
        self.feature_imgs = glob(home + "/features/*.tif")
        if self.feature_imgs == []:
            print("WARNING: empty feature image list!")
        self.randfor = RandomForestClassifier(n_estimators=50, max_depth=12, min_samples_split=5, max_features=15)
        self.X = None
        self.Y = None
        # imgname -> class -> datapoint_id -> feature_id -> float32
        # imgname -> is a dict
        # class -> is a dict
        # datapoint_id -> feature_id -> float32 is an nd
        self.data = dict()

    def build_XY(self):
        # add all the featurestacks together with appropriate labels
        X = []
        Y = []
        for img, clas in self.data.items():
            print(img)
            for i, d in clas.items():
                feats = d['features']
                print(feats)
                X.append(feats)
                Y.append(np.zeros(feats.shape[0])+i)
        self.X = np.concatenate(tuple(X), axis=0)
        self.Y = np.concatenate(tuple(Y), axis=0)

def predictions(home):
    return io.imread_collection(home + "/result_new/*.tif")

def build_data(feature_img_names, label_img_names, grayscale_img_names):
    data = dict()
    for fimg, gimg, limg in zip(grayscale_img_names, feature_img_names, label_img_names):
        stack = io.imread(fimg)
        grayimg = io.imread(gimg)
        label = io.imread(limg).astype('uint8') # TODO: remove cast and add to spec
        # add grayscale image to featurestack
        stack = np.concatenate((stack, grayimg[np.newaxis, :, :]), axis=0)

        # TODO: make sure that you don't sample more points than are available!
        # Sample from each class a certain number of times
        class_dict = dict()
        X,Y = [],[]
        for i, n_pts in enumerate([20000, 4000, 800]):
            # remember pixels used for samples in [xy]samples
            mask = label==i
            x = np.arange(mask.shape[0])
            y = np.arange(mask.shape[1])
            xx,yy = np.meshgrid(y,x)
            class_dict[i] = dict()
            class_dict[i]['samples'] = {'x' : xx[mask], 'y' : yy[mask]}

            # add that pixel + features to X
            stack_class = stack[:, mask]
            subsample = np.random.choice(stack_class.shape[1], n_pts, replace=False)
            class_dict[i]['features'] = stack_class[:, subsample].T

        data[os.path.basename(gimg)] = class_dict
    return data

def predict_Wekafeatures(greyscale_dir, feature_dir, randfor, proba=False):
    grayscale_imgs = glob(greyscale_dir + "/*.tif")
    feature_imgs = glob(feature_dir + "/*.tif")
    res = []
    for f_img, g_img in zip(feature_imgs, grayscale_imgs):
        stack = io.imread(f_img)
        grayimg  = io.imread(g_img)
        stack = np.concatenate((stack, grayimg[np.newaxis, :, :]), axis=0)
        x,y,z = stack.shape
        stack = stack.reshape((x, y*z)).T
        if proba:
            img = randfor.predict_proba(stack)
            img = img.T.reshape(3, y, z)
            dtype = 'float16'
        else:
            img = randfor.predict(stack)
            img = img.T.reshape(y, z)
            dtype = 'uint8'
        res.append(img)
        dir, base, ext = path_base_ext(f_img)
        safe_makedirs(dir + "/../result_new/")
        new_name = dir + "/../result_new/" + base + '_' + 'predict' + ext
        print("Saving to: ", new_name)
        io.imsave(new_name, img.astype(dtype))
    return res

def compare_with_existing_CRF():
    # load the pixelwise predictions from our RF and from the old CRF and the labels...
    # the problem is I don't know what score to use. I can run the dice score as a func
    # of the threshold value? That seems a reasonable way of doing things... The look at
    # the max score (min score. 0 is best).
    our_img = io.imread("./knime_test_data/data/predict/result_new10xg/features_20150127_EVLvsInner01_slice11-normalized_predict.tif")
    p0 = io.imread("./knime_test_data/data/predict/Results/PredictKNIME/predict_20150127_EVLvsInner01_slice11-normalized_level0_probs0.tif")
    p1 = io.imread("./knime_test_data/data/predict/Results/PredictKNIME/predict_20150127_EVLvsInner01_slice11-normalized_level0_probs1.tif")
    p2 = io.imread("./knime_test_data/data/predict/Results/PredictKNIME/predict_20150127_EVLvsInner01_slice11-normalized_level0_probs2.tif")
    # are there any knime predictions with ground truth?
    # can't I just use the CRF directly? Add the binary to my project and call it? On the
    # same GT labeled data?
