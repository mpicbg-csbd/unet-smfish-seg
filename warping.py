import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
plt.ion()
from scipy.misc import imresize
import skimage.transform as tform
from scipy.ndimage import label, zoom, rotate

structure = [[1,1,1], [1,1,1], [1,1,1]]

def unet_warp_orig(img, stddev=15, w=4, delta=None):
    """
    warp img according to random gaussian vector field.
    Interpolate between a square grid of w**2 random gaussian vectors with standard deviation = stddev
    """
    a,b = img.shape
    if delta:
        deltax, deltay = delta[0], delta[1]
    else:
        deltax = np.random.normal(loc=0, scale=stddev, size=(w,w))
        deltay = np.random.normal(loc=0, scale=stddev, size=(w,w))
    deltax = imresize(deltax, size=(a,b), mode='F')
    deltay = imresize(deltay, size=(a,b), mode='F')
    # MAX GRADIENTS should be less than 1 to avoid folds
    dxdx = np.max(np.diff(deltax, axis=0))
    dydx = np.max(np.diff(deltax, axis=1))
    dxdy = np.max(np.diff(deltay, axis=0))
    dydy = np.max(np.diff(deltay, axis=1))
    print("MAX GRADS", dxdx, dydx, dxdy, dydy)
    delta_big = np.stack((deltax, deltay), axis=0)
    coords = np.indices(img.shape)
    newcoords = delta_big + coords
    res = tform.warp(img, newcoords, order=1)
    return res, delta_big, coords

def plot_vector_field(img):
    """
    only designed to be the right scale for smooth warps of roughly 500^2 image patches.
    VERY FRUSTRATING THAT... images have dimensions [y,x] = [vertical,horizontal]
    and the vertical axis is always plotted DOWNWARDS!!! y goes from 0 (top) to
    y_max at the bottom! This is the opposite of all other plots...
    """
    n = 10
    res, delta, coords = unet_warp_orig(img)
    plt.figure()
    plt.imshow(img[::-1])
    plt.figure()
    plt.imshow(res[::-1])
    plt.figure()
    plt.quiver(delta[1,::n,::n],
                delta[0,::n,::n],
                headlength=2,
                headwidth=2,
                headaxislength=3)
    plt.streamplot(coords[1,0,::n]/n,
                   coords[0,::n,0]/n,
                   delta[1,::n,::n],
                   delta[0,::n,::n])
    plt.show(block=True)
    return "Awesome plot, dude!"

def warp_gaussian(img, stdev=2, w=10):
    """
    warp img according to random gaussian vector field
    w is window width.
    using w/stdev ~= 5 gives MAX GRAD approx 1.0 (and this continues linearly)
    in the unet paper they use large patches (how large?)... and a smooth warp
    field upscaled from 3x3 (roughly a factor of 100?) maybe more?... giving a 
    ratio of 100/10 ~= 10 and a max gradient of roughly 0.5... nice and smooth :)
    """
    # img[np.isnan(img)] = 0
    # assert type(w)==int
    a,b = img.shape
    da,ra = divmod(a, w)
    db,rb = divmod(b, w)
    
    deltax = np.random.normal(loc=0, scale=stdev, size=(da,db))
    deltay = np.random.normal(loc=0, scale=stdev, size=(da,db))
    deltax = imresize(deltax, size=(a,b), mode='F')
    deltay = imresize(deltay, size=(a,b), mode='F')
    dxdx = np.max(np.diff(deltax, axis=0))
    dydx = np.max(np.diff(deltax, axis=1))
    dxdy = np.max(np.diff(deltay, axis=0))
    dydy = np.max(np.diff(deltay, axis=1))
    # print("MAX GRADS", dxdx, dydx, dxdy, dydy)

    delta2 = np.stack((deltax, deltay), axis=0)
    coords = np.indices(img.shape)
    newcoords = delta2 + coords
    res = tform.warp(img, newcoords, order=3)
    return res

def warp_label_img(lab, warp_scale = 20, w = 4):
    """
    lab is a label image with 0-valued membrane
    warp a label image (i.e. a cell segmentation)
    note the label ids are NOT preserved!
    nor is the property that lab.max()==warped.max() (cells may be destroyed or created)
    """
    warpable = lab.copy()
    warpable[warpable!=0] = 1 # remove the cells
    # relabeled = label(warpable, structure=structure)[0]
    warpable = warpable.astype('float32')
    labmax = warpable.max()
    warpable /= labmax  
    deltax = np.random.normal(loc=0, scale=warp_scale, size=(w,w))
    deltay = np.random.normal(loc=0, scale=warp_scale, size=(w,w))
    warpable, delta, coords = unet_warp_orig(warpable, delta=(deltax, deltay))
    warpable *= labmax
    membrane_seg = warpable.copy()
    membrane_seg[membrane_seg > 0.75]=1
    membrane_seg[membrane_seg <= 0.75]=0
    warped_relabeled = label(membrane_seg, structure=structure)[0]
    return warped_relabeled

def random_augmentation(patch):
    """
    flip, rotate, and warp with some probability
    """
    if random.random() < 0.5:
        patch = np.flip(patch, axis=1) # axis=1 is the horizontal axis
    randangle = (random.random()-0.5)*60 # even dist between ± 30
    rotate(patch, randangle)
    if random.random() < 0.9:
        patch = warp_gaussian(patch, stdev=2, w=10) # good for full-resolution images!
    return patch

def explore_warps(img):
    # ss = np.linspace(0, 2, 8)
    ss = [2]*30
    ws = [10]
    big = np.zeros((len(ss), len(ws)) + img.shape)

    x,y = 0,0
    for s in ss:
        y=0
        for w in ws:
            # res = warp_gaussian(img, stdev=s, w=w)
            res = unet_warp_orig(img)
            big[x, y] = res
            y+=1
        x+=1
    return big

# io.imsave('big.tif', big.astype('float32'), metadata={'axes':'CTYX'}, imagej=True)

def explore_warps_multisize(img):
    """
    when you want to make a 2d tiling of images with different (but similar) sizes.
    """
    x,y = 0,0
    a,b = img.shape
    big = np.zeros(shape=(a*7, b*11))
    print(big.shape)
    # assert 0
    for s in np.linspace(0, 3, 4):
        y=0
        for w in range(5,15,3):    
            res, sumy, sumx = warp(img, mean=0, stdev=s, w=w)
            a2,b2 = res.shape
            print(a2,b2)
            print()
            big[x:x+a2, y:y+b2] = res
            y+=b
        x+=a
    return big



if __name__=='__main__':
    img500 = io.imread('img500.tif')
    plot_vector_field(img500)


