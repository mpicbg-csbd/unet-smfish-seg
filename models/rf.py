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

def balanced_sample_weights(Ytrain):
    sw = Ytrain.copy().astype('float32')
    sw[sw==0] = 1.0/len(sw[sw==0])
    sw[sw==1] = 1.0/len(sw[sw==1])
    sw[sw==2] = 1.0/len(sw[sw==2])
    sw *= len(Ytrain)
    return sw

def predict_from_stack(rafo, stack):
    a,b,c = stack.shape
    X = stack.reshape((b*c, a))
    return rafo.predict(X)

def imglist_to_XY(stacks, labels):
    if type(stacks)==np.ndarray and stacks.ndim==3:
        X,Y = to_XY_single(stacks, labels)
    elif type(stacks)==list:
        X,Y = to_XY(stacks, labels)
    else:
        raise "WTF IS WRONG WITH YOU..."
    return X,Y

def train_rafo_from_XY(X,Y, **kwargs):
    print("Class Dist: ", np.unique(Y, return_counts=True))
    # cw = {0:1, 1:10 , 2:50}
    cw = 'balanced' # 'balanced_subsample'
    rafo = xgboost.XGBClassifier()
    # rafo = sklearn.ensemble.RandomForestClassifier(n_estimators=50,
    #             max_features="sqrt", max_depth=20, min_samples_split=5,
    #             class_weight=cw)
    rafo.fit(X, Y, **kwargs)
    return rafo

def to_XY(stack, labels):
    """stack,labels are lists of featurestacks / labels. images need not be the same shape"""
    xys = map(lambda (s,l): to_XY_single(s,l), zip(stack, labels))
    def f(x1_y1, x2_y2):
        x1, y1 = x1_y1
        x2, y2 = x2_y2
        "reducing function. use on list of (X,Y) tuples to concatenate into single long X,Y."
        return (np.concatenate((x1,x2)), np.concatenate((y1,y2)))
    (X,Y) = reduce(f, xys)
    return (X,Y)

def to_XY_single(stack, lab):
    a,b,c = stack.shape
    X = stack.reshape((b*c, a))
    Y = lab.reshape(b*c)
    return (X, Y)

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
