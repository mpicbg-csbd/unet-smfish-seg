import numpy as np
import skimage.io as io

from glob import glob
import os

def train_test_split(X,Y,test_fraction):
    """
    Note: This works even when X & Y have ndim > 2
    """
    assert(0.0 < test_fraction < 0.5)
    # TODO: fix. this will break when we have more than size(int) samples
    ind = int(X.shape[0]*(1-test_fraction))
    # WARNING! IN PLACE SHUFFLE!
    np.random.shuffle(X)
    np.random.shuffle(Y)
    # Xtrain, Ytrain, Xtest, Ytest
    return X[:ind], Y[:ind], X[ind:], Y[ind:]

def subsample_ind(X,Y,test_fraction):
    """
    returns (sorted) train and test indices in appropriate ratio
    """
    assert(0.0 < test_fraction < 0.5)
    assert(X.shape[0] == Y.shape[0])
    test_size = int(Y.shape[0]*test_fraction)
    train_size = Y.shape[0]-test_size
    idxs = np.array(range(Y.shape[0]))
    test_ind = np.random.choice(Y.shape[0], size=test_size, replace=False)
    test_ind.sort()
    train_ind = filter(lambda idx: idx not in test_ind, idxs)
    train_ind = np.array(train_ind)
    # train_ind = np.random.choice(Y.shape[0], size=train_size, replace=False)
    return train_ind, test_ind

def path_base_ext(fname):
    directory, base = os.path.split(fname)
    base, ext = os.path.splitext(base)
    return directory, base, ext

def safe_makedirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def apply_operation_to_imgdir(imgdir, func, dtype='input'):
    "apply func to every img in dir, then save to new subdir."
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
