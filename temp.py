import numpy as np
import skimage.io as io
from glob import glob
import os
from __future__ import division, print_function

coll = io.imread_collection("../test_data/predict/grayscale/20150127_EVLvsInner*")
for img in coll:
    print(img.shape)
p_imgs = coll.concatenate()
print(p_imgs.shape)

coll = io.imread_collection("../test_data/train/input/labels/2015*")
for img in coll:
    print(img.shape)
t_imgs = coll.concatenate()
print(t_imgs.shape)

coll = io.imread_collection("../test_data/train/input/labels/additional-boundary-vertices/*.tif")

def fix_img(img):
    if len(img.shape)==3:
        img = np.sum(img, 2)
        img[img==65535] = 1
        img[img==131070] = 2
    print(img.shape)
    print(np.unique(img))
    return img

# for img in io.imread_collection("../test_data/train/input/labels/additional-boundary-vertices.bac/*.tif"):
# don't read images this way, they are all auto cast to the same type!

dirc = "../resources/testing_data/used_for_training_by_dagmar/all/labels/Cropped/"
dirc = "../resources/testing_data/used_for_training_by_dagmar/fold1/train/grayscale/"
dirc = "../resources/testing_data/used_for_training_by_dagmar/fold2/train/grayscale/"
for fname in glob(dirc + "*.tif"):
    print()
    # print(fname)
    img = io.imread(fname)
    # img = fix_img(img)
    print(img.shape, img.dtype)
    # print(np.unique(img, return_counts=True))
    h = np.histogram(img)
    print(h[0])
    print(h[1])
    # io.imsave("../test_data/train/input/" + os.path.basename(fname), img.astype('uint16'), compress=True)

imgs = np.array(imglist, dtype='uint8')
