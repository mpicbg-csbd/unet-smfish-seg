'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# first add the label image as a frame... then

import skimage.util as ut
import skimage.io as io

def test_model_on_folder(mod):
    greys, _ = build()
    # get a window with stride 1, predict, then patch back up into image!
    mod.predict_proba(greys)

def unwindow(img):
    """assume stride is 1"""
    a,b,c,d = img.shape
    img = np.zeros((c+a-1, d+b-1))
    for i in range(a):
        for j in range(b):
            x0 = i # *10
            y0 = j # *10
            x1 = x0+c # +28
            y1 = y0+d # +28
            img[x0:x1,y0:y1]=img[i,j]
    return img

def predict_img(img, mod):
    img2 = ut.view_as_windows(img, (28,28))
    a,b,x,y = img2.shape
    print(a,b)
    img2 = img2.reshape((a*b,x,y,1))
    y = mod.predict(img2)
    y = y.reshape(a,b,3)
    return y
    # y = unwindow(y)

def get_window_shapes(img):
    x,y = img.shape
    print("x,y: ", x, y)
    if not np.remainder(x-28,10)==0:
        print("FAIL, STRIDE DOESN'T FIT")
    if not np.remainder(y-28,10)==0:
        print("FAIL, STRIDE DOESN'T FIT")
    img = ut.view_as_windows(img, (28,28), 10)
    return img.shape


### ------------------------------------------------------------


def buildTrainingData():
    """build vector of patches from directory, then split into training and testing."""
    x,y= build()
    s1 = x.shape[0]
    print("X shape is ", x.shape)
    splt = s1//7
    X_train, X_test = x[splt:], x[:splt]
    print("Y shape is ", y.shape)
    y = y[:,14,14]
    print("Y shape is ", y.shape)
    y_train, y_test = y[splt:], y[:splt]
    # print("SHAPE", X_test.shape, X_train.shape)
    print("SHAPE", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return (X_train, y_train), (X_test, y_test)

def build():
    """build vector of patches from directory. greyscale and label"""
    greys = dir2patchvec("./knime_test_data/data/train/grayscale/grayscale_new.tif")
    labels = dir2patchvec("./knime_test_data/data/train/labels/composite/*.tif")
    labels = labels.astype('uint8')
    return greys, labels

def dir2patchvec(dir):
    """build vector of patches from directory of tifs."""
    coll = io.imread_collection(dir)
    lst = np.array(map(splt, coll))
    a,b,x,y = lst.shape
    return lst.reshape((a*b,x,y))

def splt(img):
    """split and image into a vector of patch squares."""
    x,y = img.shape
    print("x,y: ", x, y)
    if not np.remainder(x-28,10)==0:
        print("FAIL, STRIDE DOESN'T FIT")
    if not np.remainder(y-28,10)==0:
        print("FAIL, STRIDE DOESN'T FIT")
    img = ut.view_as_windows(img, (28,28), 10)
    a,b,x,y = img.shape
    return img.reshape((a*b,x,y))
