import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# plt.ion()
import sys
sys.path.insert(0, "../.local/lib/python3.5/site-packages/")
import pkg_resources
pkg_resources.require("scikit-image>=0.13.0")

import skimage
print("skimage version: ", skimage.__version__)
import skimage.io as io

import analysis
import skimage.io as io
import unet

import matplotlib.pyplot as plt
plt.ion()

def get_imglab():
        img = io.imread('data3/labeled_data_cellseg/greyscales/20150430_eif4g_dome07_slice9.tif')
        lab = io.imread('data3/labeled_data_cellseg/labels/20150430_eif4g_dome07_R3D_MASKS.tif')[0]
        lab[lab!=0]=2
        lab[lab==0]=1
        lab[lab==2]=0
        return img, lab

def small_patches(img, lab, w, dy):
        a = 1300
        b = a + dy
        img1 = img[a:a+w, a:a+w]
        lab1 = lab[a:a+w, a:a+w]
        img2 = img[b:b+w, a:a+w]
        lab2 = lab[b:b+w, a:a+w]
        return img1,img2,lab1,lab2

def predict(model, X):
        X = X[:,:,:,np.newaxis]
        res = model.predict(X, batch_size=1)
        res = res[:,:,:,1]
        return res

def test_unet():
        # prepare input
        img,lab = get_imglab()
        w = 480
        dy =20
        img1,img2,lab1,lab2 = small_patches(img, lab, w, dy)
        X = np.stack([img1, img2])
        X = unet.normalize_X(X)

        n_pool = 2
        m = unet.get_unet_n_pool(n_pool)
        m.load_weights('training/m150/unet_model_weights_checkpoint.h5')
        print(m.summary())

        Y = predict(m, X)

        itd = analysis.info_travel_dist(n_pool)
        print("ITD: ", itd)
        io.imsave('img_test.tif', X)
        io.imsave('res_test.tif', Y)
        test_XY(X,Y,itd,dy)

def test_XY(X,Y,itd,dy):
        goodimgs = X[:,itd:-itd,itd:-itd]
        goodres =  Y[:,itd:-itd,itd:-itd]
        x0 = goodimgs[0,dy:]
        x1 = goodimgs[1,:-dy]
        y0 = goodres[0,dy:] 
        y1 = goodres[1,:-dy]
        print(y0.sum(), y1.sum())
        assert np.alltrue(x0==x1)
        assert np.alltrue(y0==y1)

if __name__ == '__main__':
        # X = io.imread('img_test.tif')
        # Y = io.imread('res_test.tif')
        # test_XY(X,Y,160,20)
        test_unet()
