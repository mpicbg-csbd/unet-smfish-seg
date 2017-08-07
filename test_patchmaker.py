info = """
test patchmaker. Make sure that we can take apart and piece back together random images with perfect fidelity.
"""

import numpy as np
import matplotlib.pyplot as plt

import patchmaker

imgshape = (1075, 1075)
patch_shape = (100,100)
step = 50

img = np.indices(imgshape)/imgshape[0]*8*np.pi
img = np.cos(img[0])*np.cos(img[1])
# coords  = patchmaker.regular_patch_coords(img, patch_shape, step)
coords  = patchmaker.square_grid_coords(img, step)
patches = patchmaker.sample_patches_from_img(coords, img, patch_shape)

img2    = patchmaker.piece_together(patches, coords)
img2    = img2[:,:,0]
a,b = img.shape
a2,b2 = img2.shape
a3,b3 = min(a,a2), min(b,b2)
res     = img[:a3,:b3]==img2[:a3,:b3]
assert np.alltrue(res)

img3    = patchmaker.piece_together(patches, coords, imgshape)
img3    = img3[:,:,0]
mask = np.isnan(img3)
assert np.alltrue(img[~mask]==img3[~mask])

img4 = patchmaker.piece_together(patches, coords, imgshape, border=10)
img4 = img4[:,:,0]
mask = np.isnan(img4)
assert np.alltrue(img[~mask]==img4[~mask])

img5 = patchmaker.piece_together(patches, coords, border=10)
img5 = img5[:,:,0]
# a,b  = img.shape
# a5,b5 = img5.shape
# a6,b6 = min(a,a5), min(b,b5)
# res   = img[:a6,:b6]==img5[:a6,:b6]
# mask = np.isnan(img5)
# assert np.alltrue(img[~mask]==img5[~mask])


def plot():
	plt.imshow(img, interpolation='nearest')
	plt.figure()
	plt.imshow(img2, interpolation='nearest')
	plt.figure()
	plt.imshow(img3, interpolation='nearest')
	plt.figure()
	plt.imshow(img4, interpolation='nearest')
	plt.figure()
	plt.imshow(img5, interpolation='nearest')


thoughts1 = """
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
"""





