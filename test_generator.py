from glob import glob
import sys
import numpy as np
# sys.path.insert(0, "/home/broaddus/.local/lib/python3.5/site-packages/")
# sys.path.insert(0, "/Users/colemanbroaddus/Desktop/Projects/carine_smFISH_seg/")
# sys.path.insert(0, "../")

print('\n'.join(sys.path))

import pkg_resources
pkg_resources.require("scikit-image>=0.13.0")

import skimage
print("skimage version: ", skimage.__version__)
import skimage.io as io

import unet

imglist = [io.imread(img, plugin='tifffile') for img in glob('data3/labeled_data_cellseg/greyscales/down3x/*.tif')]
lablist = [io.imread(img, plugin='tifffile') for img in glob('data3/labeled_data_cellseg/labels/down3x/*.tif')]

for img in lablist:
	img[img!=0]=2
	img[img==0]=1
	img[img==2]=0

X,Y = unet.imglists_to_XY(imglist, lablist)

print(X.shape, Y.shape)
gen = unet.batch_generator_patches(X, Y, 100)
x,y = next(gen)
print(x.shape, y.shape)
x,y = x[:,:,:,0], y[:,:,:,1]
a,b,c = x.shape
d,b,c = y.shape
z = np.zeros((a+d,b,c))
z[::2]=x
z[1::2]=y
io.imsave('xy_gen.tif', z.astype('float32'))
