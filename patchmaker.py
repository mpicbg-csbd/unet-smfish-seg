import numpy as np


@DeprecationWarning
def sample_patches(data, patch_size, n_samples=100, verbose=False):
    """
    sample 2d patches of size patch_size from data
    """
    assert np.all([s <= d for d, s in zip(data.shape, patch_size)])
    # change filter_mask to something different if needed
    filter_mask = np.ones_like(data)
    # get the valid indices
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, data.shape)])
    valid_inds = np.where(filter_mask[border_slices])
    if len(valid_inds[0]) == 0:
        raise Exception("could not find anything to sample from...")
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    # sample
    sample_inds = np.random.randint(0, len(valid_inds[0]), n_samples)
    rand_inds = [v[sample_inds] for v in valid_inds]
    res = np.stack([data[r[0] - patch_size[0] // 2:r[0] + patch_size[0] - patch_size[0] // 2, r[1] - patch_size[1] // 2:r[1] + patch_size[1] - patch_size[1] // 2] for r in zip(*rand_inds)])
    return res

def sample_patches_from_img(coords, img, shape):
    y_width, x_width = shape
    assert coords[:,0].max() <= img.shape[0]-x_width
    assert coords[:,1].max() <= img.shape[1]-y_width
    patches = np.zeros(shape=(coords.shape[0], x_width, y_width), dtype=img.dtype)
    for m,ind in enumerate(coords):
        patches[m] = img[ind[0]:ind[0]+x_width, ind[1]:ind[1]+y_width]
    return patches

def random_patch_coords(img, n, shape):
    y_width, x_width = shape
    xc = np.random.randint(img.shape[0]-x_width, size=n)
    yc = np.random.randint(img.shape[1]-y_width, size=n)
    return np.stack((xc, yc), axis=1)

def regular_patch_coords(img, patchshape, step):
    coords = []
    dy, dx = img.shape[0]-patchshape[0], img.shape[1]-patchshape[1]
    for y in range(0,dy,step):
        for x in range(0,dx,step):
            coords.append((y,x))
    return np.array(coords)

def piece_together(patches, coords, imgshape=None, border=0):
    """
    patches must all be same shape!
    patches.shape = (sample, x, y, channel)
    coords.shape = (sample, 2 or 3 == patches.ndim-2)
    TODO: potentially add more ways of recombining than a simple average, i.e. maximum, etc
    """

    # x,y are final shape of the image
    if imgshape:
        x_size, y_size = imgshape
    else:
        x_size = coords[:,0].max() + patches.shape[0] + 1
        y_size = coords[:,1].max() + patches.shape[1] + 1
    n_samp, dx, dy, channels = patches.shape
    zeros_img = np.zeros(shape=(x_size,y_size,channels))
    count_img = np.zeros(shape=(x_size,y_size,channels))

    # ignore parts of the image with boundary effects
    mask = np.ones(patches[0].shape)
    mask[:,0:border] = 0
    mask[:,-border:] = 0
    mask[0:border,:] = 0
    mask[-border:,:] = 0

    for cord,patch in zip(coords, patches):
        x,y = cord
        zeros_img[x:x+dx, y:y+dy] += patch*mask
        count_img[x:x+dx, y:y+dy] += np.ones_like(patch)*mask

    # print(list(map(util.count_nans, [zeros_img, count_img])))
    
    res = zeros_img/count_img
    return res
