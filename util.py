import numpy as np
import skimage.io as io

from glob import glob
import os

def path_base_ext(fname):
    directory, base = os.path.split(fname)
    base, ext = os.path.splitext(base)
    return directory, base, ext

def safe_makedirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def apply_operation_to_imgdir(imgdir, func, dtype='input'):
    for file in glob(imgdir + "/*.tif"):
        img = io.imread(file)
        dir, base, ext = path_base_ext(file)
        result_img = func(img)
        # result_img = np.concatenate((img, result_img[:,:, np.newaxis]), axis=2)
        newpath = dir + os.sep + func.__name__ + os.sep
        safe_makedirs(newpath)
        new_img_name = newpath + base + ext
        print("Saving to: ", new_img_name)
        if dtype == 'input':
            io.imsave(new_img_name, result_img)
        else:
            io.imsave(new_img_name, result_img.astype(dtype))
