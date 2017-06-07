import numpy as np
import skimage.io as io

from glob import glob
import os

def sglob(string):
    return sorted(glob(string))

def subsample_ind(X,Y,test_fraction, rand_state=None):
    """
    returns (sorted) train and test indices in appropriate ratio
    """
    assert(0.0 < test_fraction < 0.5)
    assert(X.shape[0] == Y.shape[0])
    test_size = int(Y.shape[0]*test_fraction)
    train_size = Y.shape[0]-test_size
    idxs = np.array(range(Y.shape[0]))
    if rand_state is None:
        rs = np.random.RandomState()
    else:
        rs = np.random.RandomState(rand_state)
    test_ind = rs.choice(Y.shape[0], size=test_size, replace=False)
    test_ind.sort()
    # train_ind = filter(lambda idx: idx not in test_ind, idxs)
    train_ind = [idx for idx in idxs if idx not in test_ind]
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

def apply_operation_to_imgdir(imgdir, func, dtype='input', ext='.tif', inplace=False):
    "apply func to every img in dir, then save to new subdir."
    for file in glob(imgdir + "/*" + ext):
        img = io.imread(file)
        path, base, _ = path_base_ext(file)
        result_img = func(img)
        # result_img = np.concatenate((img, result_img[:,:, np.newaxis]), axis=2)
        if inplace:
            new_img_name = path + os.sep + base + ext
        else:
            newpath = path + os.sep + func.__name__ + os.sep
            safe_makedirs(newpath)
            new_img_name = newpath + base + ext
        print("Saving to: ", new_img_name)
        if dtype == 'input':
            io.imsave(new_img_name, result_img, compress=1)
        else:
            io.imsave(new_img_name, result_img.astype(dtype), compress=1)

def count_nans(img):
    return np.count_nonzero(np.isnan(img))
