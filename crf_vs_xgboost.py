"""
Compare the output of Cascaded RFs to xgboost. make jaccard score as a function of threshold
level."""

# from ipy import *

from __future__ import print_function, division

# TODO: get these things from ipy.py or __init__.py
from glob import glob
import skimage.io as io
import numpy as np

def create_result():
    """Only run once! Ever. To build the result."""
    st = rf.State("./knime_test_data/data/train/")
    st.build_data()
    st.build_XY()
    st.randfor.fit(st.X, st.Y)
    r = st.randfor
    g = "./knime_test_data/data/train/grayscale/"
    f = "./knime_test_data/data/train/features/"
    return rf.predict_Wekafeatures(g, f, r, proba=bool)



crf_mem_files = glob("./knime_test_data/data/train/PredictKNIME/grayscale_?_level1_probs1.tif")
crf_vert_files = glob("./knime_test_data/data/train/PredictKNIME/grayscale_?_level1_probs2.tif")
xgb_prob_map_files = glob("./knime_test_data/data/train/result_new/*.tif")
label_files = glob("./knime_test_data/data/train/labels/composite/*.tif")



def jaccard(img, label):
    import sklearn.metrics as m
    # TODO: the jaccard score is very high, and dominated by the background predictions...
    return m.jaccard_similarity_score(img.flatten(), label.flatten())

def run_jac():
    for a, b, c, d in zip(crf_mem_files, crf_vert_files, xgb_prob_map_files, label_files):
        # pixel values are prob of being in class 1
        rf1 = io.imread(a)
        # pixel values are prob of being in class 2
        rf2 = io.imread(b)
        # (n,m,i=1..3) pixel values are probability of being in class i
        xg  = io.imread(c)
        xg1 = xg[:,:,1]
        xg2 = xg[:,:,2]
        # pixel values are classes
        l   = io.imread(d)

        print()
        print(a)

        n = 25
        keyf = lambda t: t[0]
        maxx = lambda lizt: max(lizt, key=keyf)

        row_format ="{:>15} \t {:6.4g} \t {:5.3g}"
        # print(row_format.format("", *teams_list))
        # for team, row in zip(teams_list, data):
        #     print(row_format.format(team, *row))

        prnt = lambda x, y: print(row_format.format(x, *y))

        f = lambda x: (jaccard(rf1>x, l), x)
        prnt("CRF class1:", maxx(map(f, np.linspace(0,1,n))))
        f = lambda x: (jaccard(xg1>x, l), x)
        prnt("xgboost class1:", maxx(map(f, np.linspace(0,1,n))))
        f = lambda x: (jaccard(2*(rf2>x), l), x)
        prnt("CRF class2:", maxx(map(f, np.linspace(0,1,n))))
        f = lambda x: (jaccard(2*(xg2>x), l), x)
        prnt("xgboost class2:", maxx(map(f, np.linspace(0,1,n))))
