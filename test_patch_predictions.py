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

def get_imglab():
        img = io.imread('data3/labeled_data_cellseg/greyscales/20150430_eif4g_dome07_slice9.tif')
        lab = io.imread('data3/labeled_data_cellseg/labels/20150430_eif4g_dome07_R3D_MASKS.tif')[0]
        lab[lab!=0]=2
        lab[lab==0]=1
        lab[lab==2]=0
        return img, lab

def get_imgs(img, lab, dx):
        a = 1300
        b = a + 20
        img500 = img[a:a+dx, a:a+dx]
        lab500 = lab[a:a+dx, a:a+dx]
        img600 = img[b:b+dx, a:a+dx]
        lab600 = lab[b:b+dx, a:a+dx]
        return img500,img600,lab500,lab600

def predict():
	# prepare input
	img,lab = get_imglab()
	dx = 100
	img1,img2,lab1,lab2 = get_imgs(img, lab, dx)
	dx,dy = img1.shape
	i1 = np.reshape(img1, (dx,dy,1))
	i2 = np.reshape(img2, (dx,dy,1))
	z = np.stack((i1, i2))
	print(z.shape)

	# prepare model
	# m = unet.get_unet_7layer()
	# m.load_weights('m136/unet_model_weights_checkpoint.h5')
	m = unet.get_unet()
	m.load_weights('training/m150/unet_model_weights_checkpoint.h5')
	print(m.summary())

	# predict
	res = m.predict(z, batch_size=1)
	
        # save and compare output
	res = res[:,:,:,1]
	itd = analysis.info_travel_dist(2)
	print("ITD: ", itd)
	imgs = np.stack((img1, img2))
	io.imsave('img12.tif', imgs)
	io.imsave('res12.tif', res)
	goodimgs = imgs[:,itd:-itd,itd:-itd]
	goodres = res[:,itd:-itd,itd:-itd]
	return img1, img2, res

if __name__ == '__main__':
        predict()

# res = unet.predict_single_image(model, img500, batch_size=predict_params['batch_size'])
# combo = np.stack((img500, res), axis=0)
# plt.imshow(combo)
