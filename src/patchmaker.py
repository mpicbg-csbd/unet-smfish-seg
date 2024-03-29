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

## get patches from an image given coordinates and patch shapes.

def sample_patches_from_img(coords, img, shape, boundary_cond='mirror'):
    """
    TODO: enable boundary conditions on all sides of the img, not just bottom and right.
    """
    y_width, x_width = shape
    # assert coords[:,0].max() <= img.shape[0]-x_width
    # assert coords[:,1].max() <= img.shape[1]-y_width
    if boundary_cond=='mirror':
        a,b = img.shape
        img2 = np.zeros((2*a, 2*b), dtype=img.dtype)
        img2[:a, :b] = img.copy()
        img2[a:2*a, :b] = img[::-1,:].copy()
        img2[:a,b:2*b] = img[:,::-1].copy()
        img2[a:2*a, b:2*b] = img[::-1, ::-1].copy()
        img = img2
    patches = np.zeros(shape=(coords.shape[0], x_width, y_width), dtype=img.dtype)
    for m,ind in enumerate(coords):
        patches[m] = img[ind[0]:ind[0]+x_width, ind[1]:ind[1]+y_width]
    return patches

## Different ways of sampling pixel coordinates from an image

def random_patch_coords(img, n, shape):
    y_width, x_width = shape
    xc = np.random.randint(img.shape[0]-x_width, size=n)
    yc = np.random.randint(img.shape[1]-y_width, size=n)
    return np.stack((xc, yc), axis=1)

## deprecated because we don't want coordinates to depend on patchshape
@DeprecationWarning
def regular_patch_coords(img, patchshape, step):
    coords = []
    dy, dx = img.shape[0]-patchshape[0], img.shape[1]-patchshape[1]
    for y in range(0,dy,step):
        for x in range(0,dx,step):
            coords.append((y,x))
    return np.array(coords)

def square_grid_coords(img, step):
    a,b = img.shape
    a2,ar = divmod(a, step)
    b2,br = divmod(b, step)
    a2 += 1
    b2 += 1
    ind = np.indices((a2, b2))
    ind *= step
    ind = np.reshape(ind, (2, a2*b2))
    ind = np.transpose(ind)
    return ind

## piece together a single image from a list of coordinates and patches

def piece_together(patches, coords, imgshape=None, border=0):
    """
    patches must all be same shape!
    patches.shape = (sample, x, y, channel) or (sample, x, y)
    coords.shape = (sample, 2)
    TODO: potentially add more ways of recombining than a simple average, i.e. maximum, etc
    """

    if patches.ndim == 3:
        patches = patches[:,:,:,np.newaxis]
    n_samp, dx, dy, channels = patches.shape
    
    x_size = coords[:,0].max() + dx
    y_size = coords[:,1].max() + dy
    if imgshape:
        x_host, y_host = imgshape
        x_size, y_size = max(x_size, x_host), max(y_size, y_host)
    patch_img = np.zeros(shape=(x_size, y_size,channels))
    count_img = np.zeros(shape=(x_size, y_size,channels))

    # ignore parts of the image with boundary effects
    mask = np.ones((dx, dy, channels))
    if border>0:
        mask[:,0:border] = 0
        mask[:,-border:] = 0
        mask[0:border,:] = 0
        mask[-border:,:] = 0

    for cord, patch in zip(coords, patches):
        x,y = cord
        patch_img[x:x+dx, y:y+dy] += patch*mask
        count_img[x:x+dx, y:y+dy] += np.ones_like(patch)*mask

    # if imgshape:
    #     a,b = imgshape
    #     patch_img = patch_img[:a,:b]
    #     count_img = count_img[:a,:b]
    
    res = patch_img/count_img
    if imgshape:
        a,b = imgshape
        res = res[:a, :b]
    return res



    # # x_size = coords[:,0].max() + dx
    # # y_size = coords[:,1].max() + dy
    # if imgshape:
    #     x_host, y_host = imgshape
    #     x_size, y_size = max(x_size, x_host), max(y_size, y_host)

    # count_img = np.zeros(shape=(x_size, y_size))

