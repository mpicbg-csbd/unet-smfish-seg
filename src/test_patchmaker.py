info = """
test patchmaker. Make sure that we can take apart and piece back together random images with perfect fidelity.
"""

import numpy as np
# import matplotlib.pyplot as plt
# plt.ion()
import skimage.io as io

import patchmaker

def make_img_patches_coords(img_shape, patch_shape, step):
	img = np.indices(img_shape)/img_shape[0]*8*np.pi
	img = np.cos(img[0])*np.cos(img[1]/4)
	coords  = patchmaker.square_grid_coords(img, step)
	patches = patchmaker.sample_patches_from_img(coords, img, patch_shape)
	return img, patches, coords

def make_img_patches_coords_random(img_shape, patch_shape, step):
	img   = np.random.randint(0,10,img_shape)
	# img = np.cos(img[0])*np.cos(img[1]/4)
	coords  = patchmaker.square_grid_coords(img, step)
	patches = patchmaker.sample_patches_from_img(coords, img, patch_shape)
	return img, patches, coords

def test_img_patches_coords(img, patches, coords, border):
	res = patchmaker.piece_together(patches, coords, imgshape=img.shape, border=border)
	res = res[:,:,0]
	b = 198
	print("res Test:", np.alltrue(img[b:-b,b:-b]==res[b:-b,b:-b]))
	img = np.stack([img, res]).astype('float32')
	io.imsave('img.tif', img)
	diff = img[0,b:-b,b:-b] - img[1,b:-b,b:-b]
	print(diff.max(), diff.min())
	io.imsave('diff.tif',diff)

def setborder(img, border, value):
	if border>0:
	    img[:,0:border] = value
	    img[:,-border:] = value
	    img[0:border,:] = value
	    img[-border:,:] = value
	return img



def test1():
	img_shape = (1075, 1075)
	patch_shape = (100,100)
	step = 50
	img, patches, coords = make_img_patches_coords(img_shape, patch_shape, step)
	plot = test_img_patches_coords(img, patches, coords, 0)
	plot()

def test2():
	"""
	Test image invariance wrt border operations. No matter what we do to the border, the result doesn't care!
	"""
	# img_shape = (2857, 6271)
	img_shape = np.random.randint(2400, 3600, (2,))
	patch_shape = (480, 480)
	border = 77
	step = 360 - 2*border
	img, patches, coords = make_img_patches_coords(img_shape, patch_shape, step)
	patches = [setborder(patch, border, 0) for patch in patches]
	patches = np.array(patches)
	test_img_patches_coords(img, patches, coords, border=border)

def test3():
	"""
	Requires more testing. Works as long as step < x_size, y_size
	"""
	img_shape = np.random.randint(400,700,(2))
	patch_shape = np.random.randint(10,20,(2))
	# border = np.random.randint(0,30)
	border = 0
	# step = patch_shape[0] - 2*border
	# step = np.random.randint(0,30)
	step=patch_shape.min()-1
	print(img_shape, patch_shape, border, step)
	img, patches, coords = make_img_patches_coords_random(img_shape, patch_shape, step)
	plot = test_img_patches_coords(img, patches, coords, border)
	plot()

def test4():
	"""
	Just a check to see if we calculated coords, border, etc wrong in patchmaker. We didn't, but we still have square artifacts. This
	means there must be a problem with our calculation of border=itd based on the unet. test2 confirms this without using Unet module, 
	and test5 confirms this while using the Unet module to chop up X.
	"""
	img = io.imread('data3/labeled_data_cellseg/greyscales/20150127_EVLvsInner01_slice11.tif')
	img_shape = img.shape
	patch_shape = (480, 480)
	border = 92
	step = 480 - 2*border
	coords = patchmaker.square_grid_coords(img, step)
	patches = io.imread('imgs/Ypred.tif')
	print(patches.shape)
	for border in [90, 91, 92, 93, 94, 95]:
		res = patchmaker.piece_together(patches, coords, imgshape=img.shape, border=border)
		res = res[:,:,0]
		io.imsave('img_{:03d}.tif'.format(border), res.astype('float32'))
	# plot = test_img_patches_coords(img, patches, coords, border=border)
	# plot()

def test5():
	"""
	This proves that the chopping up and putting-back-together routines in patchmaker
	are correct! Even after undoing the image-wise normalization and removing the boundary, the images are pixelwise identical.
	They work with border==100, but when applying the Unet, the Y_images still has vertical and horizontal line artifacts. WHY!?!?!
	Even if our ITD was off by a small amount....
	"""
	img = io.imread('imgs/20150127_EVLvsInner01_slice11_predict_x_100.tif')
	img1, img2 = img[0], img[1]
	print(img1.shape, img2.shape)
	border = 100
	img1 = img1[border:,border:]
	img2 = img2[border:,border:]
	img1 -= img1.min()
	img2 -= img2.min()
	img1 /= img1.max()
	img2 /= img2.max()
	res = img1-img2
	print(res.max())
	io.imsave('img1.tif', img1)
	io.imsave('img2.tif', img2)

def test6():
	"""
	This shows
	1. Border size does have an effect on the output, even when border size is greater than the ITD
	2. The differences between the images seem to be stronger where there is signal in the image.
	"""
	img1 = io.imread('imgs/20150127_EVLvsInner01_slice11_predict_90.tif')
	img2 = io.imread('imgs/20150127_EVLvsInner01_slice11_predict_94.tif')
	img3 = io.imread('imgs/20150127_EVLvsInner01_slice11_predict_100.tif')
	bmax = 100
	img1 = img1[1, bmax:,bmax:]
	img2 = img2[1, bmax:,bmax:]
	img3 = img3[1, bmax:,bmax:]
	# print((img2-img1).max())
	plt.figure(figsize=(20,14))
	plt.imshow((img2-img1))
	# print((img3-img2).max())
	plt.figure(figsize=(20,14))
	plt.imshow((img3-img2))
	# print((img3-img1).max())
	plt.figure(figsize=(20,14))
	plt.imshow((img3-img1))

def test7():
	"""
	Is it deterministic? Yes!?
	Maybe not! Uwe tried many patches and found a small error.
	"""
	img1 = io.imread('imgs/20150127_EVLvsInner01_slice11_predict_100.tif')
	img2 = io.imread('imgs/determ_test1.tif')
	img3 = io.imread('imgs/determ_test2.tif')
	img4 = io.imread('imgs/determ_test_cpu.tif')
	border = 100
	img1 = img1[1, border:, border:]
	img2 = img2[1, border:, border:]
	img3 = img3[1, border:, border:]
	img4 = img4[1, border:, border:]
	# print((img2-img3).max())
	# print((img1-img3).max())
	# print((img2-img1).max())
	print((img4-img1).max())
	# print((img4-img2).max())
	# print((img4-img3).max())
	plt.figure(figsize=(20,14))
	plt.imshow((img4-img1))
	# print((img3-img2).max())
	# plt.figure(figsize=(20,14))
	# plt.imshow((img4-img2))
	# print((img3-img1).max())
	# plt.figure(figsize=(20,14))
	# plt.imshow((img4-img3))


# if __name__ == '__main__':
# 	test1()
# 	test2()
# 	test3()
# 	test4()


thoughts1 = """
----
Mon Aug  7 10:00:00 2017 UTC
----

- How are we going to get rid of the black borders on the edges of the results (img2, img3, img4)?
- The Ronneberger paper does it by introducing mirroring at the borders. We can do the same.

Should we be allowed to specify an imgshape separately from the coordinates and patch sizes in piece_together?
What should we do with the "extra space". Should they be a value? no. they should be nan! should we return nan or
ask for one of a few simple ways of dealing with nan? Better to return nan... The problem of mirroring the image 
during patch creation should be solved *before* we get here. We allow overlapping patches and combine them via averaging.

We don't need to solve such a genral problem just yet. We're only using patchmaker in unet.py. And we only need to test that 
the images are the same in the non-nan regions, and everywhere once we use the mirroring boundary conditions.

After adding in the mirroring boundary conditions, the assert img=img2 no longer makes sense.
Changed the comparison to be only on the valid domain.

Now, what do we do when we specify imgshape, but our patches don't fit in evenly? Cut them off!
Works now.

NOTE: setting border to nonzero in piece_together gives us a border on the upper and left sides, but not the bottom in img4.
This is because the nan border in img4 is cut off at the end because of our reshaping. Let's try with no imgshape and see...

OK, patchmaker looks good :)

Testing unet revealed that our method for fixing the size of the resulting image patch is wrong. It prevents the patch from being too large,
but doesn't fix it if it's too small... Why would our patch be too small? Let's test the exact case from the unet... 
size from ... data3/labeled_data_cellseg/greyscales/20150128_fig10_slice10.tif

Fixed the undersize problem in piece_together.

NOTE: strange thing... random images of randing(0,10,(largeshape)) don't look random.

More testing. step has to be <= patch_shape.min() or the first assert fails.

---- 
Tue Aug  8 13:31:32 2017 UTC
----

Let's check that patchmaker is working a appropriately in unet. We shouldn't see any grey areas except a border-width nan on the top and left,
and a variable [0, border-width] nan value on the bottom and right. and really we should be sampling patches s.t. we don't worry about top and left either.

Using a different step with precomupted Ypred.tif doesn't work, because Ypred depends on step length, which depends on border width.

The only remaining possibility is that we're computing ITD wrong, or misunderstand ITD concept.

"""





