from glob import glob
import sys
sys.path.insert(0, "/home/broaddus/.local/lib/python3.5/site-packages/")
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

# import warping
# import random
# i = random.randint(0, Y.shape[0])
# warping.plot_vector_field(Y[i])

print(X.shape, Y.shape)
gen = unet.batch_generator_patches(X, Y, 100)
x,y = next(gen)
print(x.shape, y.shape)
