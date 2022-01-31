"""
Take an image, return a featurestack."""

# from ipy import *

from __future__ import print_function, division

# TODO: get these things from ipy.py or __init__.py
from glob import glob
import skimage.io as io
import numpy as np
import skimage.filters as flt
import skimage.feature as fea

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

goodnames = ['sobel', 'roberts_neg_diag', 'scharr_v', 'sobel_h', 'scharr_h',
            'sobel_v', 'prewitt_v', 'prewitt', 'canny', 'prewitt_h', 'roberts',
            'scharr', 'laplace', 'roberts_pos_diag']


def advanced(img):
    for i in np.linspace(0.1,.6, 7):
        for th in np.linspace(0, 3.1415/2.0, 4):
            io.imsave("gabor/img{}_{}.tif".format(i,th), flt.gabor(img, frequency=i, theta=th)[0])

def gabor_stack(img):
    fqs = np.linspace(0.1,.6, 10)
    ths = np.linspace(0, 3.1415/2.0, 5)
    stack = [flt.gabor(img, frequency=fq, theta=th)[0] for fq in fqs for th in ths]
    return np.array([img] + stack)

def stack_features(img):
    """FIX: set of features to stack is determined at load time for this module."""
    img /= img.max()
    funclist = [(name, f) for (name, f) in flt.__dict__.iteritems() if callable(f) if name not in notgoodnames]
    print([name for (name, f) in funclist])
    app = lambda (name, func): func(img)
    stack = [img] + map(app, funclist)
    return np.array(stack)

if __name__ == '__main__':
    import sys
    img = io.imread(sys.argv[1])
    io.imsave(sys.argv[1][:-4] + "_fstack.tif", stack_features(img).astype('float32'))
