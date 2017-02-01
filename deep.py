from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from glob import glob

import skimage.io as io
import skimage.feature as skf
import skimage.feature.blob as B
import skimage.filters as F

import keras as ks

import os

class State:
    def __init__(self, home):
        self.home = home
        # self.home = "../knime_test_data/data/train/"
        self.grayscale_imgs = glob(home + "/grayscale/*.tif")
        self.label_imgs = glob(home + "/labels/composite/*.tif")
        self.feature_imgs = glob(home + "/features/*.tif")
        # self.randfor = RandomForestClassifier(n_estimators=50, max_depth=12, min_samples_split=5, max_features=15)
        self.X = None
        self.Y = None
        self.xsamples = None
        self.ysamples = None
        self.stacks = []
        self.labels = []

    def add_stackAndlabels(this):
        for ig, gy, lb in zip(this.feature_imgs, this.grayscale_imgs, this.label_imgs):
            stack = io.imread(ig)
            grayimg = io.imread(gy)
            label = io.imread(lb).astype('uint8')
            # add grayscale image to featurestack
            stack = np.concatenate((stack, grayimg[np.newaxis, :, :]), axis=0)
            this.stacks.append(stack)
            this.labels.append(label)

    def train_first_level(this):
        # add each grayscale image to the top of the stack of features
        # NOTE: we don't check to make sure the features are correct! We just put them in.
        # We'll do a thorough checking that featurestacks adhere to a spec before we send them in.
        # NOTE: The different featurestacks *do not have to be the same size!* Feel free to
        # crop your different images in a way that makes sense for each.

        X,Y = [],[]
        for ig, gy, lb in zip(this.feature_imgs, this.grayscale_imgs, this.label_imgs):
            stack = io.imread(ig)
            grayimg = io.imread(gy)
            label = io.imread(lb).astype('uint8')
            # add grayscale image to featurestack
            stack = np.concatenate((stack, grayimg[np.newaxis, :, :]), axis=0)

            # TODO: make sure that you don't sample more points than are available!
            # Sample from each class a certain number of times

            xsamples = []
            ysamples = []
            for i, n_pts in enumerate([20000, 4000, 800]):
                # remember pixels used for samples in [xy]samples
                mask = label==i
                x = np.arange(mask.shape[0])
                y = np.arange(mask.shape[1])
                xx,yy = np.meshgrid(y,x)
                xsamples.append(xx[mask])
                ysamples.append(yy[mask])

                # add that pixel + features to X
                stack_class = stack[:, mask]
                print(stack_class.shape)
                subsample = np.random.choice(stack_class.shape[1], n_pts, replace=False)
                X.append(stack_class[:, subsample].T)
                Y.append(np.zeros(n_pts)+i)

        this.xsamples = np.concatenate(tuple(xsamples), axis=0)
        this.ysamples = np.concatenate(tuple(ysamples), axis=0)
        X = np.concatenate(tuple(X), axis=0)
        Y = np.concatenate(tuple(Y), axis=0)
        this.randfor.fit(X, Y)
        this.X = X
        this.Y = Y
