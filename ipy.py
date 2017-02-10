"""An interactive module. Use me by copy and pasting lines into iPython!"""

# global python
from __future__ import print_function, division
from glob import glob

# global scikit
import numpy as np
import skimage.io as io
import timeit

# local
# import keras_classifier as kc
# import mnist_keras as mk
import rf
import build_featurestack as bf
import sklearn

# work with a single image
img = io.imread("./knime_test_data/data/train/grayscale/grayscale_0.tif")
print(img.shape)
lab = io.imread("./knime_test_data/data/train/labels/composite/vertex_labels_0.tif")
gaborstack = bf.gabor_stack(img)
wekastack = io.imread("./knime_test_data/data/train/features/features_0.tif")
wekastack = np.concatenate((img[np.newaxis,:,:], wekastack), axis=0)

a,b,c = wekastack.shape
subsample = np.random.choice(b*c, size=None, replace=False)

# train on subset and predict on self
rafo = rf.train_rafo_from_stack(wekastack, lab, subsample=subsample)
ypred = rf.predict_from_stack(rafo, wekastack)
print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))


# run training across a set of images
img_coll = io.imread_collection("./knime_test_data/data/train/grayscale/grayscale_?.tif")
lab_coll = io.imread_collection("./knime_test_data/data/train/labels/composite/vertex_labels_?.tif")
stacks = map(bf.gabor_stack, img_coll)
stacks = np.array(stacks)
X,Y = rf.


st = rf.State("./knime_test_data/data/train/")
st.build_data()
st.build_XY()
st.randfor.fit(st.X, st.Y)

import sklearn

mod = mk.trainmodel()
td = kc.buildTrainingData()
