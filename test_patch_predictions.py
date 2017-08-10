import sys
sys.path.insert(0, "../.local/lib/python3.5/site-packages/")
import pkg_resources
pkg_resources.require("scikit-image>=0.13.0")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# plt.ion()

import skimage
print("skimage version: ", skimage.__version__)
import skimage.io as io

import analysis
import skimage.io as io
import unet

import matplotlib.pyplot as plt
import patchmaker
plt.ion()

def get_imglab():
    img = io.imread('data3/labeled_data_cellseg/greyscales/20150430_eif4g_dome07_slice9.tif')
    lab = io.imread('data3/labeled_data_cellseg/labels/20150430_eif4g_dome07_R3D_MASKS.tif')[0]
    lab[lab!=0]=2
    lab[lab==0]=1
    lab[lab==2]=0
    return img, lab

def predict(model, X):
    X = X[:,:,:,np.newaxis]
    res = model.predict(X, batch_size=1)
    res = res[:,:,:,1]
    return res

def test_unet():
    # prepare input
    img,lab = get_imglab()
    img = img-img.min()
    img = img/img.max()
    w  = 480
    yinds = np.arange(10)*4 + 1100
    xinds = [1100]*10
    coords = np.array(list(zip(yinds, xinds)))
    X = patchmaker.sample_patches_from_img(coords, img, (w,w))
    # X = unet.normalize_X(X)

    n_pool = 2
    m = unet.get_unet_n_pool(n_pool)
    m.load_weights('training/m150/unet_model_weights_checkpoint.h5')
    print(m.summary())

    Y = predict(m, X)

    itd = analysis.info_travel_dist(n_pool)
    print("ITD: ", itd)
    io.imsave('img_test.tif', X)
    io.imsave('res_test.tif', Y)
    # test_XY(X,Y,itd,dy)

def test_XY(X,Y,itd,dy):
    """
    despite removing the invalid regions, we are still left
    with many unexplained discrepencies between the two predictions
    """
    validx = X[:,itd:-itd,itd:-itd]
    validy  = Y[:,itd:-itd,itd:-itd]
    x0 = validx[0,dy:]
    x1 = validx[1,:-dy]
    y0 = validy[0,dy:]
    y1 = validy[1,:-dy]
    io.imsave("xvalid.tif", x0-x1)
    io.imsave("yvalid.tif", y0-y1)
    print(y0.sum(), y1.sum())
    print("X1==X2? : ", np.alltrue(x0==x1))
    print("Y1==Y2? : ", np.alltrue(y0==y1))        
    # so it's not all true... now what
    mask = y0!=y1
    res = y1-y0
    inds = np.indices(mask.shape)
    inds = inds[:,mask]
    print(inds[0].min(), inds[0].max(), inds[1].min(), inds[1].max())
    hist_unique(inds[0])
    hist_unique(inds[1])
    # hist_unique(res)
    dwindling_histogram(res, 10)
    print("Number of discrepencies:", mask.sum())
    io.imsave('ydiffs.tif', mask.astype('uint8'))

def dwindling_histogram(array, topn):
    uniqs, counts = np.unique(array, return_counts=True)
    fig = plt.figure()
    for _ in range(topn):
        ind = np.argmax(counts)
        print("Mode:", uniqs[ind], counts[ind])
        plt.clf()
        plt.plot(uniqs, counts, '.')
        fig.canvas.draw()
        counts[ind] = 0
        input("continue?")

def hist_unique(array):
    vals, counts = np.unique(array, return_counts=True)
    plt.plot(vals, counts, '.', label = str(counts.sum()))
    plt.legend()

def run():
    X = io.imread('img_test.tif')
    Y = io.imread('res_test.tif')
    test_XY(X,Y,20,20)

if __name__ == '__main__':
    run()

