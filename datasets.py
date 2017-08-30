import os
from glob import glob
import skimage.io as io
import numpy as np
import util
import patchmaker
import skimage.exposure as expo

def sglob(string):
    return sorted(glob(string))

knime_train_data_greys_bgblack         = lambda : sglob("data/knime_test_data/data/train/greyscale_bg_removed/bg_removed?.tif")
knime_train_data_greys                 = lambda : sglob("data/knime_test_data/data/train/grayscale/grayscale_?.tif")
knime_train_data_labels                = lambda : sglob("data/knime_test_data/data/train/labels/composite/vertex_labels_?.tif")
knime_train_data_keras_mem_predictions = lambda : sglob("data/grayscale_?_predict.tif")
knime_predict_data_greys               = lambda : sglob("data/knime_test_data/data/predict/grayscale/*.tif")

unseen_greys         = lambda : sglob("data/unseen_greys/mean8/*.tif")
unseen_labels        = lambda : sglob("data/unseen_labels/pooled/*.tif")
# unseen_mem_predict = lambda : sglob("data/unseen_mem_predict/*.tif")
unseen_mem_predict   = lambda : sglob("data/2015*predict.tif")
unseen_seg           = lambda : sglob("data/2015*seg.tif")

# --- data2 directory

greyscales   = lambda : sglob("data2/greyscales/*.tif")
labels       = lambda : sglob("data2/Cell_segmentations_paper/*.tif")

greyscales3x = lambda : sglob("data2/greyscales/down3x/*.tif")
labels3x     = lambda : sglob("data2/labels/down3x/*.tif")

greyscales6x = lambda : sglob("data2/greyscales/down6x/*.tif")
labels6x     = lambda : sglob("data2/labels/down6x/*.tif")

# --- data3 directory

# Here we've split the data into cell segmentation data and membrane prob map data and segregated by downscaling size

# greyscale images with membrane labelings (by hand)
mem_images         = lambda : sglob("data3/labeled_data_membranes/images/*.tif")
mem_images_small3x = lambda : sglob("data3/labeled_data_membranes/images/small3x/*.tif")
mem_images_big     = lambda : sglob("data3/labeled_data_membranes/images_big/*.tif/")
mem_images_big2x   = lambda : sglob("data3/labeled_data_membranes/images_big/smaller2x/*.tif")
mem_images_big6x   = lambda : sglob("data3/labeled_data_membranes/images_big/smaller6x/*.tif")

# label images drawn by hand on cell boundaries
mem_labels         = lambda : sglob("data3/labeled_data_membranes/labels/*.tif")
mem_labels_small3x = lambda : sglob("data3/labeled_data_membranes/labels/small3x/*.tif")
mem_labels_big     = lambda : sglob("data3/labeled_data_membranes/labels_big/*.tif")
mem_labels_big2x   = lambda : sglob("data3/labeled_data_membranes/labels_big/smaller2x/*.tif")
mem_labels_big6x   = lambda : sglob("data3/labeled_data_membranes/labels_big/smaller6x/*.tif")

# greyscale and label images resulting from cell segmentation pipeline
seg_images         = lambda : sglob("data3/labeled_data_cellseg/greyscales/*.tif")
seg_images_small3x = lambda : sglob("data3/labeled_data_cellseg/greyscales/down3x/*.tif")
seg_images_small6x = lambda : sglob("data3/labeled_data_cellseg/greyscales/down6x/*.tif")
seg_labels         = lambda : sglob("data3/labeled_data_cellseg/labels/*.tif")
seg_labels_small3x = lambda : sglob("data3/labeled_data_cellseg/labels/down3x/*.tif")
seg_labels_small6x = lambda : sglob("data3/labeled_data_cellseg/labels/down6x/*.tif")

# EXTRA DATA: full-size greyscale and label images annotated by cell segmentation pipeline 
seg_images_extra = sglob("data3/labeled_data_cellseg/greyscales/*timeseries*/*.tif")


def split_in_half_for_train_test(grey_imgs, label_imgs):
    print("CREATING NDARRAY PATCHES")
    grey_leftside   = []
    label_leftside  = []
    grey_rightside  = []
    label_rightside = []
    for grey,lab in zip(grey_imgs, label_imgs):
        a,b = grey.shape
        print("Shape of img:")
        print(a,b)
        grey_leftside.append(grey[:,0:b//2])
        label_leftside.append(lab[:,0:b//2])
        grey_rightside.append(grey[:,b//2:])
        label_rightside.append(lab[:,b//2:])
    print("SPLIT INTO TRAIN AND TEST")
    print("WE HAVE TO SPLIT THE IMAGES IN HALF FIRST, OTHERWISE THE VALIDATION DATA WILL STILL BE PRESENT IN THE TRAINING DATA, BECAUSE OF OVERLAP.")
    X_train,Y_train = imglists_to_XY(grey_leftside, label_leftside)
    print("X_train.shape = ", X_train.shape, " and Y_train.shape = ", Y_train.shape)
    X_vali, Y_vali  = imglists_to_XY(grey_rightside, label_rightside)
    return X_train,Y_train,X_vali,Y_vali

def build_stakk():
    greys  = sglob("/Volumes/Coleman_Pocket/Carine_project/data3/labeled_data_membranes/images_big/smaller2x/*.tif")
    labels = sglob("/Volumes/Coleman_Pocket/Carine_project/data3/labeled_data_membranes/labels_big/smaller2x/*.tif")
    count = 0
    end = None
    step = 256
    width = 256 # must be factor of 2^d, d=n_maxpooling (across all models!)
    stakk = []
    sizes = []
    for a,b in zip(greys[:end], labels[:end]):
        img = io.imread(a)
        lab = io.imread(b)
        # lab = lab[1]

        ## normalize each image to [0,1]. Don't get rid of bright outliers!
        img = expo.equalize_adapthist(img, [10,10])
        # img = img.astype('float32')
        # img -= img.min()
        # img /= img.max()

        sizes.append(img.shape)
        coords = patchmaker.square_grid_coords(img, step)
        img_pat = patchmaker.sample_patches_from_img(coords, img, (width, width))
        
        ## normalize each X patch
        # img_pat -= img_pat.min(axis=(1,2), keepdims=True)
        # img_pat = img_pat.astype('uint16')
        # img_pat *= (2**16-1)//img_pat.max(axis=(1,2), keepdims=True)
        
        lab_pat = patchmaker.sample_patches_from_img(coords, lab, (width, width))
        patches = np.stack([img_pat, lab_pat], axis=1)
        stakk.append(patches)
        count += 1
    stakk = np.concatenate(stakk, axis=0)
    return stakk, sizes

def get_all_big_tifs(basedir):
    count = 0
    tiflist = []
    for a,b,c in os.walk(basedir):
        for imgName in c:
            if imgName.endswith(".tif") and 'down' not in a:
                if 'timeseries' in imgName and 'MembraneMiddle' in imgName:
                    if 'zoomed' in imgName:
                        count += 1
                        d_name = os.path.join(a, imgName)
                        print(d_name)
                        tiflist.append(d_name)
                else:
                    count += 1
                    d_name = os.path.join(a, imgName)
                    print(d_name)
                    tiflist.append(d_name)
    print(count)
    return tiflist

@DeprecationWarning
def imsave(fname, img, axes='TYXC', **kwargs):
    print("works!")
    io.imsave(fname, img, compress=6, plugin='tifffile', metadata={'axes':axes}, imagej=True, **kwargs)

@DeprecationWarning
def imread(fname, **kwargs):
    return io.imread(fname, plugin='tifffile', **kwargs)


def norm_img(img):
    mn,mx = img.min(), img.max()+1e-10
    img -= mn
    img = img/(mx-mn)
    return img


## ---- HOW WE MADE THE DATA

def make_prediction_overlays():
    pairs = zip(unseen_seg(), unseen_labels(), unseen_greys(), unseen_seg_files())
    def save(tup):
        a,b,c, name = tup
        path, base, ext = util.path_base_ext(name)
        new = np.stack((c,a,b), axis=0)
        imsave(base + "_overlay" + ext, new)
    map(save, pairs)

def max_pool_downscale():
    from skimage.util import view_as_windows
    def pooled3(img):
        # img[img==0] = 2000 # so that membrane is 1 is the min value
        print("shape: ", img.shape)
        img = view_as_windows(img, 6, step=6)
        img = np.max(img, axis=(2,3))
        # now permute back
        # img[img==2000] = 0
        return img
    util.apply_operation_to_imgdir("data2/labels/", pooled3)

def mean_downscale():
    from skimage.util import view_as_windows
    def down3x(img):
        s = img.shape
        print("shape: ", img.shape)
        # if s[0] > s[1]:
        # img = np.transpose(img)
        img = view_as_windows(img, 3, step=3)
        img = np.mean(img, axis=(2,3)).astype(np.float32)
        return img/img.max()
    util.apply_operation_to_imgdir("data2/greyscales/", down3x)

def strip_channel():
	from util import path_base_ext
	greyscales = sglob("data2/greyscales/*.tif")
	labels = sglob("data2/Cell_segmentations_paper/*.tif")
	data = zip(greyscales, labels)
	for g, l in data:
		img1 = imread(g)
		img2 = imread(l)
		path,base,ext = path_base_ext(g)
		img1 = img1[np.newaxis, :, :]
		res = np.concatenate((img1, img2), axis=0)
		res = np.transpose(res, (1,2,0))
		print(img1.shape, img2.shape)
		print(img1.dtype, img2.dtype)
		# res = np.stack((img1, img2), axis=0)
		# assert 1<0
		imsave("data2/labels/" + base+ext, res)

def relabeled():
	from util import path_base_ext
	greyscales = sglob("data2/greyscales/*.tif")
	labels = sglob("data2/Cell_segmentations_paper/*.tif")
	data = zip(greyscales, labels)
	for g, l in data:
		img2 = imread(l)
		path,base,ext = path_base_ext(g)
		img2 = img2[0,:,:]
		img2[img2!=0] = 3
		img2[img2==0] = 1
		img2[img2==3] = 0
		imsave("data2/combined/" + base+ext, img2)


# def rotate():
#     def rot(img):
#         s = img.shape
#         if s[0] > s[1]:
#             img = np.rot90(img, 3)
#         img -= img.min()
#         img *= 20
#         return img
#     ipy.util.apply_operation_to_imgdir("unseen_greys/", rot)


# In the end we had to rotate manually, because the different images were all
# rotated/flipped differently.
