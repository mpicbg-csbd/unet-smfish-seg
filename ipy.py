"""An interactive module. Use me by copy and pasting lines into iPython!"""

# local
import keras_classifier as kc
import mnist_keras as mk
import rf

# global python
from __future__ import print_function, division
from glob import glob

# global scikit
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt



img = io.imread("./knime_test_data/data/train/grayscale/grayscale_0.tif")
print(img.shape)

st = rf.State("./knime_test_data/data/train/")
st.build_data()
st.build_XY()
st.randfor.fit(st.X, st.Y)



mod = mk.trainmodel()
td = kc.buildTrainingData()
