from glob import glob
import numpy as np

import pkg_resources
pkg_resources.require("scikit-image>=0.13.0")

import skimage
print("skimage version: ", skimage.__version__)
import skimage.io as io

# import unet

imglist = [io.imread(img, plugin='tifffile') for img in glob('data3/labeled_data_cellseg/greyscales/down3x/*.tif')]
lablist = [io.imread(img, plugin='tifffile') for img in glob('data3/labeled_data_cellseg/labels/down3x/*.tif')]

img = io.imread('data3/labeled_data_cellseg/greyscales/20150430_eif4g_dome07_slice9.tif')
lab = io.imread('data3/labeled_data_cellseg/labels/20150430_eif4g_dome07_R3D_MASKS.tif')[0]

lab[lab!=0]=2
lab[lab==0]=1
lab[lab==2]=0

img500 = img[1000:1500, 1000:1500]
lab500 = lab[1000:1500, 1000:1500]

import warping

io.imsave('xy.tif', np.stack((img500, lab500)).astype('float32'))
x,y = warping.randomly_augment_patches(img500, lab500, False, 10, 0)
io.imsave('xy_warped.tif', np.stack((x,y)).astype('float32'))