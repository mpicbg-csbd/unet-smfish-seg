import unet
from glob import glob

imglist = [unet.io.imread(img) for img in glob('data3/labeled_data_cellseg/greyscales/down3x/*.tif')]
lablist = [unet.io.imread(img) for img in glob('data3/labeled_data_cellseg/labels/down3x/*.tif')]

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