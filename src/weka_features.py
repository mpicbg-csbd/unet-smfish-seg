"""Build A Random Forest from the weka featurestacks. Run predictions on weka featurestacks."""

from glob import glob
import util
import skimage.io as io
import numpy as np

def predict_Wekafeatures(greyscale_dir, feature_dir, randfor, proba=False):
    grayscale_imgs = glob(greyscale_dir + "/*.tif")
    feature_imgs = glob(feature_dir + "/*.tif")
    res = []
    for f_img, g_img in zip(feature_imgs, grayscale_imgs):
        stack = io.imread(f_img)
        grayimg  = io.imread(g_img)
        stack = np.concatenate((stack, grayimg[np.newaxis, :, :]), axis=0)
        x,y,z = stack.shape
        stack = stack.reshape((x, y*z)).T
        if proba:
            img = randfor.predict_proba(stack)
            img = img.T.reshape(3, y, z)
            dtype = 'float16'
        else:
            img = randfor.predict(stack)
            img = img.T.reshape(y, z)
            dtype = 'uint8'
        res.append(img)
        dir, base, ext = util.path_base_ext(f_img)
        safe_makedirs(dir + "/../result_new/")
        new_name = dir + "/../result_new/" + base + '_' + 'predict' + ext
        print("Saving to: ", new_name)
        io.imsave(new_name, img.astype(dtype))
    return res
