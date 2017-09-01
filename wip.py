import pandas as pd
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
plt.ion()
import scipy.ndimage as ndimage
import skimage.exposure as exposure
from glob import glob
import subprocess

import summarize_models
import datasets
import segtools
import warping
import unet


def load_state():
    images = glob('training/m343/*.tif')
    labels = glob('/Volumes/Coleman_Pocket/Carine_project/data3/labeled_data_membranes/labels_big/smaller2x/*.tif')
    img,pred = io.imread(images[4])
    lab = io.imread(labels[4])
    mask = np.isnan(pred)
    pred[mask] = 0

    state = {'img_names': images, 
             'lab_names': labels,
             'img': img, 
             'pred': pred,
             'lab': lab,
            }
    return state

def a(pred):
    from skimage.feature import peak_local_max
    pred[pred < 0.75]=0
    local_maxi = peak_local_max(1-pred, min_distance=20, threshold_abs=0.5, indices=False, ) #footprint=np.ones((10, 10)))

    qsave(50*local_maxi.astype('uint8'))

def qsave(img):
    print(img.min(), img.max())
    io.imsave('qsave.tif', img)
    subprocess.call("open qsave.tif", shell=True)

def summary():
    if 'df' not in globals():
        df = pd.read_pickle('summary.pkl')
        df = summarize_models.update_df(df)
    # print(df.columns)
    df = summarize_models.get_n_best(df[df.n_patches > 20], 30)

def idea1():
    idea = """
    Remove crap from segmented image by only keeping the largest connected component of foreground pixels.
    """
    ids,cts = np.unique(seg,return_counts=True)
    order = np.argsort(cts)[::-1]
    seg_clean = seg==ids[order][1]

def idea2():
    idea = """
    Try different normalization schemes, including adaptive local histogram equalization.
    """
    imglist = []
    for d in [20,40,80]:
        imglist.append(exposure.equalize_adapthist(grey, [d,d]))

goals = """
We want the best view of our results overlayed with the original image and the ground truth.
We want to be able to see where our model disagrees with the ground truth and compare the two.
We want to be able to edit our solution easily. We also want to be able to change the original labeling and retrain on better ground truth. 
"""

ideas = """
GALA: http://journal.frontiersin.org/article/10.3389/fninf.2014.00034/full#B5

Which says:
GALA belongs a class of segmentation algorithms called agglomerative algorithms, in which segments are formed by merging smaller segments.
Other examples include mean agglomeration (Arbeláez et al., 2010), the graphical models of Andres et al.
(2012a,b), and Learning to Agglomerate Superpixel Hierarchies (LASH) (Jain et al., 2011), which is most similar to GALA.

We should use something like hierarchical clustering to find the optimal segmentation. It could be learned...
Would these techniques work when our cell segmentation is NOT a segmentation of the full image, because it ignores boundaries?

Start off with an initial oversegmentation. Which we get with watershed on the probability maps? Seed the watershed by getting local minima of 
probabilities?...
"""

def compute_uncertainty(img_names, dirs):
    # sumimg = 
    import util, os
    for imgname in img_names:
        path, base, ext = util.path_base_ext(imgname)
        predictions = []
        for d in dirs:
            imgname = d + '/' + base + ext
            if os.path.exists(imgname):
                img, pred = io.imread(imgname)
                predictions.append(pred)
        predictions = np.stack(predictions)
        img_mean = np.mean(predictions, axis=0)
        img_std = np.std(predictions, axis=0)
        img_min = np.min(predictions, axis=0)
        img_min_thresh = img_min > 0.02
        img_max = np.max(predictions, axis=0)
        qsave(np.stack([img, img_mean, img_std, img_min, img_min_thresh, img_max]))
        qsave(np.stack([img, img_min]))


def img_to_levels(img, levels=[0.5]):
    """
    split a continuous value image into discrete levels with uint values
    """
    img = img/img.max()
    levels = [0] + levels + [1.0]
    print(levels)
    count = 0
    n = len(levels)
    img2 = img.copy().astype('uint8')
    for i in range(n-1):
        mi = levels[i]
        ma = levels[i+1]
        mask = (mi < img) & (img < ma)
        img2[mask] = i
    return img2

def show_mistakes(state):
    """
    This shows off the human annotation mistakes clearly! That's great, and means the model is generalizing nicely.
    """
    img  = state['img']
    lab  = state['lab']
    pred = state['pred']
    
    pred = pred>0.9
    pred_backup = pred.copy()
    
    # mask_1 = pred < 0.05
    # mask_2 = (0.05 < pred) & (pred < 0.95)
    # mask_3 = 0.95 < pred
    # pred[mask_1] = 0
    # pred[mask_2] = 1
    # pred[mask_3] = 2
    # res = np.stack([img, pred*100])
    # # qsave(res)

    seg, _ = ndimage.label(pred<0.95)

    lab[lab==2] = 1
    pred = pred_backup.copy()
    mask_lab = 0.8 < pred
    pred[mask_lab==lab]  = 0
    pred[mask_lab > lab] = 0
    pred[lab > mask_lab] = 1

    res = np.stack([img/img.max(), pred, mask_lab]).astype('float16') #, pred, pred])
    qsave(res)

    img2 = img.copy()
    ma = np.percentile(img, 99)
    print(ma)
    img2[img2 < ma] = 0
    # qsave(img2)


problem = """
MY MODELS DON'T CONVERGE [EVERY TIME]
This is really two problems:
1. When it fails to converge, what is happening?
2. Why does this happen to me and Laurent, but not to Martin and Uwe? What is different about our problems?
Secondary Problems:
1. Why does it happen in streaks? Becuase I vary my hyperparams one-at-a-time?

Hypothesis:
- FALSE! Convergence is a very sensitive function of our hyperparams...

Test:
Train the same model multiple times. Convergence is random.

Hypotheses:
1. It depends on initial conditions.
2. It depends on the trajectory through param space, and therefore on sort order of X_train.
But for both causes, it's unclear *how* the learning gets stuck.

Hypothesis: 
- We are losing the gradient info early on in the model [during first epoch, and for ALL SAMPLES] so params don't get updated.
- Alternative: We are updating params & moving around in param space, but we're lost in a region without
any good local minima. This is consistent with hypothesis 1, but not with hypothesis 2.

This means we can rule out all of the following as causes, although they may still influence the rate.
- Label noise
- Bad normalization
And label noise is not even really a problem with the hand-labeled data.

Tests:
- We can prevent X_train from randomly sorting at the beginning of every epoch. Then we'll be able to say if
learning depends on initialization or not.
- We can also fix the initialized params somehow, then we'll be able to say if our assumptions are true.
Find a set of params that seems to converge about half to the time. Fix these params and use them for testing.
NOTE: In the Ronneberger paper they use a very particular initialization scheme with weights coming from a centered normal
dist with stdev = sqrt(2/fan_in) where fan_in is the number of incoming neurons.

Test:
I've changed the normalization, to min&max [0,1] imagewise. And the percentage of models that converge has gone way down.
After further testing even the small difference between global [0,1] normalization and image-wise [0,1] normalization has the effect
of dramatically changing my convergence rate. Let's load each image and get the min and max. I expect the max value to change significantly if
this difference in normalization is to have any effect... It's true. The max values are outliers and vary wildly per image. We want to do a different,
smarter kind of normalization. Also, after going back to patch-wise [0,1] normalization my models converge really well and look good!
Now I can ignore the convergence problem and the normalization problem and just always do patchwise normalization. This destroys the most info,
but somehow works the best...
"""

question = """
What's the right normalization?
Clearly the normalization is important, because our model can't learn for some of them, and can for others...
Our normalization should be based on the histograms, which clearly show the various classes of pixels present in the image.
- Histogram eqalization flattens the histogram. This is not what we want. Also, in general, histogram flattening is not possible, because we alreay have multiple counts of the same value at the pixel level.
- Quantile based normalization just reassigns the nth-highest value in the (image? histogram?) to the nth-highest value from a reference distribution.
    - But the number of pixels that have that value will not, in general, be the same.
    - And there is still some ambiguity when the set of values in image1 is greater than image2.
- We do NOT want either of these techniques. We want our histogram to *not* be flat! We just want to shift the peaks of the modes s.t. they are aligned
across images...
- RELATED: What is the method for auto contrast adjustment used by Fiji?
- RELATED: Signal to noise ratio is calculable given our segmentations/labelings. What is signal to noise ratio? For classification problems it's a measure
    of how easy the problem is if you could only use the histograms. 
- RELATED: We can do a simple histogram-based classification.
- RELATED: Someone should do a review of all the kinds of normalization, and when they're useful... whitening, Batch Normalization (Ioffe and Szegedy, 2014), etc
- Image normalization vs Deep Learning normalization vs general ML normalization (quantile based, etc)? Standard normalization is optimized for image viewing...
"""

hypothesis = """
I want to tell if it's really the normalization which has caused all my current difficulties...
clip to 1,99 percentile | minor trouble, but messes up bright regions
stakk to 0,1 | lots of trouble
image to 0,1 | lots of trouble
image Expo.equalize_adapthist | lotsoftrouble
patchwise to 0,1 | super easy, everything converges
log transform | ???
"""

def hist():
    """
    Explore the image histograms as a function of class id.
    """
    greys = sorted(glob("/Volumes/Coleman_Pocket/Carine_project/data3/labeled_data_membranes/images_big/smaller2x/*.tif"))
    labs  = sorted(glob("/Volumes/Coleman_Pocket/Carine_project/data3/labeled_data_membranes/labels_big/smaller2x/*.tif"))
    id_list = []
    count_list = []
    for i in range(len(greys))[:3]:
        img = io.imread(greys[i])
        lab = io.imread(labs[i])
        print(img.min(), img.max(), img.dtype)
        print(lab.min(), lab.max(), lab.dtype)
        # plt.hist(img.flatten(), bins=1000, histtype='step', label=str(i), normed=True, cumulative=True)
        color = tuple(np.random.rand(3))
        scale = np.percentile(img, 99)
        for j in [0,1,2]:
            ids, cts = np.unique(img[lab==j], return_counts=True)
            id_list.append(ids)
            count_list.append(ids)
            # qsave(img)
            plt.plot(ids/scale, np.log10(cts), '.', color=color, label='img={} lab={}'.format(i,j))
    plt.legend()
    return id_list, count_list



def distance_transform():
    dis = ndimage.distance_transform_edt(1-lab)
    dis[dis>20]=20

