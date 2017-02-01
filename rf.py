from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from glob import glob

import skimage.io as io
import skimage.feature as skf
import skimage.feature.blob as B
import skimage.filters as F

import os

from sklearn.ensemble import RandomForestClassifier

# FIXME: These lists should be combined.

dont_use_list = ['vsobel', 'hsobel', 'vscharr', 'vprewitt', 'hprewitt', 'roberts_positive_diagonal', 'roberts_negative_diagonal',
    'hscharr', 'deprecated', 'copy_func', 'LPIFilter2D', 'gabor_filter', 'gabor_kernel', 'gabor']

dontusenames = ['median',
 'gaussian',
 'wiener',
 'threshold_yen',
 'threshold_adaptive',
 'rank_order',
 'inverse',
 'gaussian_filter',
 'threshold_otsu',
 'gabor',
 'threshold_li',
 'threshold_isodata']

notgoodnames = dont_use_list + dontusenames

def path_base_ext(fname):
    directory, base = os.path.split(fname)
    base, ext = os.path.splitext(base)
    return directory, base, ext

def example1():
    # [io.imsave(name, f(img)) for f in dir(F)]

    fname = "../knime_test_data/data/train/grayscale/grayscale_0.tif"
    dir, base, ext = path_base_ext(fname)
    img = io.imread(fname)
    img /= img.max()

    # I want to make a feature stack. But I run into errors. To get around this I need
    # to manually select which filters I use and how I call them...

    # If you don't want to import the module...
    from inspect import getmembers, isfunction
    functions_list = [o for o in getmembers(F) if isfunction(o[1]) if o[1] != 'copy_func']

    funclist = [(name, f) for (name, f) in F.__dict__.iteritems() if callable(f) if name not in notgoodnames]

    for name, f in funclist:
        new_name = "../resources/" + base + '_' + name + ext
        try:
            io.imsave(new_name, f(img).astype('float32'))
        except (TypeError, ValueError, AttributeError, AssertionError) as e:
            print(); print();
            print(new_name)
            dontusenames.append(name)
            print(type(e), e)

class State:
    def __init__(self, home):
        self.home = home
        # self.home = "../knime_test_data/data/train/"
        self.grayscale_imgs = glob(home + "/grayscale/*.tif")
        self.label_imgs = glob(home + "/labels/composite/*.tif")
        self.feature_imgs = glob(home + "/features/*.tif")
        if self.feature_imgs == []:
            print("WARNING: empty feature image list!")
        self.randfor = RandomForestClassifier(n_estimators=50, max_depth=12, min_samples_split=5, max_features=15)
        self.X = None
        self.Y = None
        # imgname -> class -> datapoint_id -> feature_id -> float32
        # imgname -> is a dict
        # class -> is a dict
        # datapoint_id -> feature_id -> float32 is an nd
        self.data = dict()

    def build_data(self):
        for fimg, gimg, limg in zip(self.feature_imgs, self.grayscale_imgs, self.label_imgs):
            stack = io.imread(fimg)
            grayimg = io.imread(gimg)
            label = io.imread(limg).astype('uint8') # TODO: remove cast and add to spec
            # add grayscale image to featurestack
            stack = np.concatenate((stack, grayimg[np.newaxis, :, :]), axis=0)

            # TODO: make sure that you don't sample more points than are available!
            # Sample from each class a certain number of times
            class_dict = dict()
            X,Y = [],[]
            for i, n_pts in enumerate([20000, 4000, 800]):
                # remember pixels used for samples in [xy]samples
                mask = label==i
                x = np.arange(mask.shape[0])
                y = np.arange(mask.shape[1])
                xx,yy = np.meshgrid(y,x)
                class_dict[i] = dict()
                class_dict[i]['samples'] = {'x' : xx[mask], 'y' : yy[mask]}

                # add that pixel + features to X
                stack_class = stack[:, mask]
                subsample = np.random.choice(stack_class.shape[1], n_pts, replace=False)
                class_dict[i]['features'] = stack_class[:, subsample].T

            self.data[os.path.basename(gimg)] = class_dict

    def build_XY(self):
        # add all the featurestacks together with appropriate labels
        X = []
        Y = []
        for img, clas in self.data.items():
            print(img)
            for i, d in clas.items():
                feats = d['features']
                print(feats)
                X.append(feats)
                Y.append(np.zeros(feats.shape[0])+i)
        self.X = np.concatenate(tuple(X), axis=0)
        self.Y = np.concatenate(tuple(Y), axis=0)

def predictions(home):
    return io.imread_collection(home + "/result_new/*.tif")

def apply_operation_to_imgdir(imgdir, func, dtype='uint16'):
    for file in glob(imgdir + "/*.tif"):
        img = io.imread(file)
        dir, base, ext = path_base_ext(file)
        result_img = func(img)
        result_img = np.concatenate((img, result_img[:,:, np.newaxis]), axis=2)
        newpath = dir + os.sep + func.__name__ + os.sep
        safe_makedirs(newpath)
        new_img_name = newpath + base + ext
        print("Saving to: ", new_img_name)
        io.imsave(new_img_name, result_img.astype(dtype))

def predict_Wekafeatures(home, randfor, proba=False):
    grayscale_imgs = glob(home + "/grayscale/*.tif")
    feature_imgs = glob(home + "/features/*.tif")
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
        dir, base, ext = path_base_ext(f_img)
        safe_makedirs(dir + "/../result_new/")
        new_name = dir + "/../result_new/" + base + '_' + 'predict' + ext
        print("Saving to: ", new_name)
        io.imsave(new_name, img.astype(dtype))
    return res

def safe_makedirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def param_search_through_forests():
    RandomForestClassifier()

def check_train(path):
    """The images contained in path must adhere to a strict spec to be used for training.
    This functions checks them against that spec."""

    # must have a `train/` directory with labeled images and featuremaps...
    # feature images should all be of the same type? (Can we make random forests with a
    # mix of numerical and label info?) IF they are not all float images, then we can't
    # use the same set of classifiers/methods. How do we know when an pixel represents a
    # label vs an intensity? Even intensities can be multiple types! floats, ints and
    # uints!

    # tif only has one 'f'

    # The names in the various directories must all align. And the images must be the
    # same size if they share a name.

    # Images *should* have metadata like voxel size and microscopy info. We should be able
    # to include this information in our classifiers. If we want to include this data in
    # our tifs, then it must *also* have its own spec. (essentially a dictionary spec)

    # featurestacks should include the greyscale image. *What about data augmentation?*

    # for classification lables, we should check to make sure that the labels are all
    # the same value and that the values appear in roughly the same distribution!

    # intensity images should all be normalized in the same way. Sometimes float images
    # are forced to take on values in the range [-1, 1] (like in scikit-image!)

    # what about image metadata being stored in filenames? This is obnoxious, but common
    # and useful for quick, but it does tend to blow up path names and seems like a good
    # way to introduce nasty characters into paths... (like spaces!)

    # filenames must be made up of a set of cross-platform standard ascii characters and
    # no spaces, dashes(or?) or slashes. Just upper and lowercase letters, numbers, '_' and '.'.

    # When working with new data, either we edit it s.t. it conforms to our spec, or we
    # change our code s.t. it can read the new data. If you change the data, then your
    # data doesn't look like the original (which is still alive, on someone else's machine
    # but if you change your code, then you have some extra piece of code which has to
    # live in your project made just for reading the new stuff and conforming it to the
    # spec every time you want to load it and run your code. Which way is better? Aha! If
    # you plan on distributing your code, then you can expect that the users will be able
    # to conform their data to your spec, but not that they will be able to add code to
    # read it! This puts the onus on *them* to make your tool work, but shows them exactly
    # how to do it. Putting constraints/expectations in a spec program is *better* than
    # hiding it away in the docs. Users will misread or just not read your docs.
    # Having an (interactive) spec checker will force them to
    # deal with their issues!

    # Number of features in feature stack must be the same across whole directory.


def blob():
    # TODO
    # The filters used on images allows us to do a pixel-wise classification of images, but
    # there are also more coarse-grained features we might want to use... Why not build a
    # global forest which includes the output from skimage.feature as well as filters?
    # How would we incorporate these global properties into our pixel-patch classification
    # decisions in a smart way?

    [B.blob_doh, B.blob_log, B.blob_dog, B.arccos, B.gaussian_filter, B.gaussian_laplace, B.hypot, B.integral_image]
