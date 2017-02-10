"""Here's where we build and control our random forests"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from sklearn.ensemble import RandomForestClassifier
import sklearn
import xgboost

from glob import glob
import os


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

    def build_data(self):
        self.data = build_data(self.feature_imgs, self.label_imgs, self.grayscale_imgs)

def predict_from_stack(rafo, stack):
    X = to_X_single(stack)
    return rafo.predict(X)



def train_rafo_from_stack(stack, labels, subsample=None):
    if type(stack)==np.ndarray:
        X,Y = to_XY_single(stack, labels)
    elif type(stack)==list:
        X,Y = to_XY(stack, labels)
    else:
        raise "WTF IS WRONG WITH YOU..."
    # X,Y = subsample_XY(X,Y,size=3000)
    if subsample != None:
        X,Y = X[subsample, :], Y[subsample]
    print("Class Dist: ", np.unique(Y, return_counts=True))
    # cw = {0:1, 1:10 , 2:50}
    cw = 'balanced_subsample'
    # rafo = xgboost.XGBClassifier(n_estimators=50, max_depth=20, min_samples_split=5, class_weight=cw)
    rafo = sklearn.ensemble.RandomForestClassifier(n_estimators=50, max_features="sqrt", max_depth=None, min_samples_split=5, class_weight=cw)
    rafo.fit(X, Y)
    return rafo

def to_XY(stack, labels):
    """stack,labels are lists of featurestacks / labels. images need not be the same shape"""
    xys = map(lambda (s,l): to_XY_single(s,l), zip(stack, labels))
    def f((x1,y1), (x2,y2)):
        (np.stack(x1,x2), np.stack(y1,y2))
    (X,Y) = reduce(f, xys)
    return (X,Y)

def to_XY_single(stack, lab):
    a,b,c = stack.shape
    X = stack.reshape((b*c, a))
    Y = lab.reshape(b*c)
    return (X, Y)

def to_X_single(stack):
    a,b,c = stack.shape
    X = stack.reshape((b*c, a))
    return X

def subsample_XY(X,Y, size=10000):
    ss = np.random.choice(len(Y), size=size, replace=False)
    return (X[ss,:], Y[ss])

def build_data(feature_img_names, label_img_names, grayscale_img_names):
    """load all featurestacks and labels into dict."""
    data = dict()
    for fimg, gimg, limg in zip(feature_img_names, grayscale_img_names, label_img_names):
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
