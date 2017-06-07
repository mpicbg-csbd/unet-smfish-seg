from glob import glob
import skimage.io as io
import numpy as np
import util

def sglob(string):
    return sorted(glob(string))

knime_train_data_greys_bgblack = lambda : sglob("data/knime_test_data/data/train/greyscale_bg_removed/bg_removed?.tif")
knime_train_data_greys = lambda : sglob("data/knime_test_data/data/train/grayscale/grayscale_?.tif")
knime_train_data_labels = lambda : sglob("data/knime_test_data/data/train/labels/composite/vertex_labels_?.tif")
knime_train_data_keras_mem_predictions = lambda : sglob("data/grayscale_?_predict.tif")
knime_predict_data_greys = lambda : sglob("data/knime_test_data/data/predict/grayscale/*.tif")

unseen_greys = lambda : sglob("data/unseen_greys/mean8/*.tif")
unseen_labels = lambda : sglob("data/unseen_labels/pooled/*.tif")
# unseen_mem_predict = lambda : sglob("data/unseen_mem_predict/*.tif")
unseen_mem_predict = lambda : sglob("data/2015*predict.tif")
unseen_seg = lambda : sglob("data/2015*seg.tif")

# --- data2 directory

greyscales = lambda : sglob("data2/greyscales/*.tif")
labels = lambda : sglob("data2/Cell_segmentations_paper/*.tif")

greyscales3x = lambda : sglob("data2/greyscales/down3x/*.tif")
labels3x = lambda : sglob("data2/labels/down3x/*.tif")

greyscales6x = lambda : sglob("data2/greyscales/down6x/*.tif")
labels6x = lambda : sglob("data2/labels/down6x/*.tif")

# newgreys = lambda : sglob("data2/20150513_New_data/*.tif")
# greyscales_down3x = lambda : sglob("data2/labeled_data_100xObj/images/down3x/*.tif")
# labels_down3x = lambda : sglob("data2/labeled_data_100xObj/labels/pooled/*.tif")

# --- data3 directory

# Here we've split the data into cell segmentation data and membrane prob map data


def imsave(fname, img, **kwargs):
    io.imsave(fname, img, compress=6, plugin='tifffile', **kwargs)

def imread(fname, **kwargs):
    return io.imread(fname, plugin='tifffile', **kwargs)

# ---- HOW WE MADE THE DATA

# def make_prediction_overlays():
#     pairs = zip(unseen_seg(), unseen_labels(), unseen_greys(), unseen_seg_files())
#     def save((a,b,c, name)):
#         path, base, ext = util.path_base_ext(name)
#         new = np.stack((c,a,b), axis=0)
#         imsave(base + "_overlay" + ext, new)
#     map(save, pairs)

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
