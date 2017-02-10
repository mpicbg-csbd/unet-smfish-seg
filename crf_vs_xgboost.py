"""
Compare the output of Cascaded RFs to xgboost. make jaccard score as a function of threshold level."""

# from ipy import *

from __future__ import print_function, division

# TODO: get these things from ipy.py or __init__.py
from glob import glob
import skimage.io as io
import numpy as np
import weka_features as wekaf

def create_result():
    """Only run once! Ever. To build the result."""
    st = rf.State("./knime_test_data/data/train/")
    st.build_data()
    st.build_XY()
    st.randfor.fit(st.X, st.Y)
    r = st.randfor
    g = "./knime_test_data/data/train/grayscale/"
    f = "./knime_test_data/data/train/features/"
    return wekaf.predict_Wekafeatures(g, f, r, proba=bool)

# load everything
crf_mem_files = glob("./knime_test_data/data/train/PredictKNIME/grayscale_?_level1_probs1.tif")
crf_vert_files = glob("./knime_test_data/data/train/PredictKNIME/grayscale_?_level1_probs2.tif")
xgb_prob_map_files = glob("./knime_test_data/data/train/result_new/*.tif")
label_files = glob("./knime_test_data/data/train/labels/composite/*.tif")
crf1 = map(io.imread, crf_mem_files)
crf2 = map(io.imread, crf_vert_files)
xgbs = map(io.imread, xgb_prob_map_files)
labs = map(io.imread, label_files)
crfs = zip(crf1, crf2)
def joint_crf((crf1, crf2)):
    """How should we interpret the set of two binary classifications
    as a 3-class probability? NOTE: `new` must be normalized over 3rd dimension!"""
    x,y = crf1.shape
    new = np.zeros((x,y,3), dtype=crf1.dtype)
    new[:,:,0] = 1.0 - crf1 - crf2
    new[:,:,1] = crf1 # - crf2
    new[:,:,2] = crf2
    return new
crfs = map(joint_crf, crfs)
xgbs = np.array(xgbs)
labs = np.array(labs)
crfs = np.array(crfs)

# compute the confusion_matrix
import sklearn.metrics as m
xgbs_l = xgbs.argmax(3)
crfs_l = crfs.argmax(3)
a,b,c = labs.shape
print("XGBOOST Confusion Matrix")
xgbs_confusion = m.confusion_matrix(labs.flatten(), xgbs_l.flatten())
print(xgbs_confusion)
print("CRF Confusion Matrix")
crfs_confusion = m.confusion_matrix(labs.flatten(), crfs_l.flatten())
print(crfs_confusion)
