doc = """
# Module for Label images

We want to think about some tests, and perhaps a spec for labeled images. They
should be uint type?
The labeled images that Carine made have zero-valued membranes which separate the
>= 2 valued cells and the 1-valued background. But we don't *need* to have this
boundary layer, and it probably also shouldn't count towards our cell-matching score. (Neither should the background?).
"""

# import networkx as nx
import numpy as np
import skimage.io as io
import colorsys
from numba import jit

# or get it from scipy.ndimage.morphology import generate_binary_structure
structure = [[1,1,1], [1,1,1], [1,1,1]] # this is the structure that was used by Benoit & Carine!

## JUST COLORING STUFF

def pastel_colors_RGB(n_colors=10, brightness=0.5, value=0.5):
    """
    a cyclic map of equal brightness and value. Good for elements of an unordered set.
    """
    HSV_tuples = [(x * 1.0 / n_colors, brightness, value) for x in range(n_colors)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return RGB_tuples

def pastel_colors_RGB_gap(n_colors=10, brightness=0.5, value=0.5):
    """
    leaves a gap in Hue, so colors don't cycle around, but go from Red to Blue
    """
    HSV_tuples = [(x * 0.75 / n_colors, brightness, value) for x in range(n_colors)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return RGB_tuples

def label_colors(bg_ID=1, membrane_ID=0, n_colors = 10, maxlabel=1000):
    RGB_tuples = pastel_colors_RGB(n_colors=10)
    # intens *= 2**16/intens.max()
    assert membrane_ID != bg_ID
    RGB_tuples *= maxlabel
    RGB_tuples[membrane_ID] = (0, 0, 0)
    RGB_tuples[bg_ID] = (0.01, 0.01, 0.01)
    return RGB_tuples

def labelImg_to_rgb(img, bg_ID=1, membrane_ID=0):
    """
    TODO: merge this with the numba version from cell_tracker
    """
    # TODO: the RGB_tuples list we generate is 10 times longer than it needs to be
    RGB_tuples = label_colors(bg_ID, membrane_ID, n_colors=10, maxlabel=img.max())
    a,b = img.shape
    rgb = np.zeros((a,b,3), dtype=np.float32)
    for val in np.unique(img):
        mask = img==val
        print(mask.shape)
        # rgb[mask,:] = np.array(get_color_from_label(val))
        rgb[mask,:] = RGB_tuples[val]
    # f16max = np.finfo(np.float16).
    print(rgb.max())
    # rgb *= 255*255
    return rgb.astype(np.float32) # Preview on Mac only works with 32bit or lower :)

# MISC

@jit
def permute_img(img, perm=None):
    """
    Permute the labels on a labeled image according to `perm`, if `perm` not given
    then permute them randomly.
    Returns a copy of `img`.
    """
    print("ok")
    if not perm:
        perm = np.arange(img.max()+1)
        np.random.shuffle(perm)
    res = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i,j] = perm[img[i,j]]
    return res

def permutation_from_matching(matching):
    ar = np.arange(matching.shape[0])
    p1 = np.argmax(matching, axis=1)
    # p1 is *almost* the permutation we want...
    # what do we do if matching[i,j]=1, but matching[j,:] = all zeros ?
    # which label do we give to j in perm? a new, biggest label?
    # yep, that's what we'll do...
    # this way the intersection of labels in the two images are just the ones in the matching!
    perm = np.where(matching[ar,p1]!=0, p1, -1)
    s = perm[perm==-1].shape[0]
    perm[perm==-1] = np.arange(s)+perm.max()+1
    return perm

# @jit
# def matching_from_permutation(perm):
#     matching = np.zeros(perm[0])

# Segmentation LOSSES, ERRORS, SCORES, MATCHINGS, GRAPHS

@jit
def pixel_sharing_graph(img1, img2):
    """
    returns an ndarray representing a bipartite graph with pixel overlap count as the edge weight.
    img1 and img2 must be same shape, and label (uint) images.
    """
    imgs = np.stack((img1, img2), axis=2)
    mat = np.zeros((img1.max()+1, img2.max()+1), dtype=np.uint32)
    a,b,c = imgs.shape
    for i in range(a):
        for j in range(b):
            mat[imgs[i,j,0], imgs[i,j,1]] += 1
    return mat

def matching_overlap(mat, fraction=0.5):
    """
    create a matching given pixel_sharing_graph of two label images based on mutually overlapping regions of sufficient size.
    NOTE: a true matching is only gauranteed for fraction > 0.5. Otherwise some cells might have deg=2 or more.
    NOTE: doesn't break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
    """
    matc = mat / np.sum(mat, axis=1, keepdims=True)
    matr = mat / np.sum(mat, axis=0, keepdims=True)
    matc50 = matc > fraction
    matr50 = matr > fraction
    result = matc50 * matr50
    return result.astype('uint8')

def matching_max(mat):
    """
    matching based on most overlapping pixels
    """
    rowmax = np.argmax(mat, axis=0)
    colmax = np.argmax(mat, axis=1)
    starting_index = np.arange(len(rowmax))
    equal_matches = colmax[rowmax[starting_index]]==starting_index
    rm, cm = rowmax[equal_matches], colmax[rowmax[equal_matches]]
    matching = np.zeros_like(mat)
    matching[rm, cm] = 1
    return matching

def is_matching(mat):
    assert mat.dtype in [np.bool, np.uint8, np.uint16, np.uint32, np.uint64]
    assert np.sum(mat,0).max() == np.sum(mat,1).max() <= 1
    return True

def intersection_over_union(mat):
    rsum = np.sum(mat,0, keepdims=True)
    csum = np.sum(mat,1, keepdims=True)
    return mat / (rsum + csum - mat)

def seg(mat):
    """
    calculate seg from pixel_sharing_graph
    seg is the average conditional-iou across ground truth cells
    conditional-iou gives zero if not in matching
    ----
    calculate conditional intersection over union (CIoU) from matching & pixel_sharing_graph
    for a fraction > 0.5 matching. Any CIoU will be > 1/3. But there may be some
    IoU as low as 1/2 that don't match, and thus have CIoU = 0.
    """
    matching = matching_overlap(mat, fraction=0.5)
    iou = intersection_over_union(mat)
    conditional_iou = matching * iou
    seg = np.max(conditional_iou, axis=1)
    seg = np.mean(seg)
    return seg

def matching_score(matching):
    print("{} matches out of {} GT objects and {} predicted objects.".format(matching.sum(), matching.shape[0], matching.shape[1]))


# ----------------------------------------------------------------------

@DeprecationWarning
def permlabels(img, perm=None):
    """
    Permute the labels on a labeled image according to `perm`, if `perm` not given
    then permute them randomly.
    Returns a copy of `img`.
    """
    m = img.max()
    if perm is None:
        perm = np.random.permutation(m+1)
    img2 = img.copy()
    for i in range(m+1):
        img2[img==i] = perm[i]
    return img2

@DeprecationWarning
def _matching_matrix_slow(img1, img2):
    "img1 and img2 must be same shape, and label (uint) images."
    imgs = np.stack((img1, img2), axis=2)
    mat = np.zeros((img1.max()+1, img2.max()+1), dtype=np.uint32)
    a,b,c = imgs.shape
    for edg in imgs.reshape((a*b,c)):
        mat[edg[0], edg[1]] += 1
    return mat

@DeprecationWarning
def old_50percent_criterion():
    gt_size = np.sum(mat[row,:]) # an int
    max_match_ind = np.argmax(mat[row,:])
    intersection = mat[row, max_match_ind]
    if 2*intersection > gt_size:
        # we have at least a 50% match!
        match_size = np.sum(mat[:,max_match_ind])
        jac = intersection / (gt_size + match_size - intersection)
        return jac
    else:
        return 0

@DeprecationWarning
def match_score_1(mat):
    """
    Compute the matching score from image1 to image2. (2D only)
    First build adjacency matrix between labels in img1 and img2.
    Then, for each label in img1, find most common matching label `l2` in img2.
    Do the same for `l2` matching into img1.
    If the labels agree, then add one to score.
    TODO: make sure that this score is img1-img2 symmetric.
    YES, it is symmetric. For Each vtx in bipartite id graph draw a single edge to it's most frequent match. We just count the number of 2-cycles. This is independent of where you start counting.
    IDEA:
    Faster label-image-comparison by first concatenating id1 and id2 (at same pixel location) into a single number (bitwise?) and then making a histogram, and then undoing the histogram back into a directed graph.
    """
    # TODO: not 50%, but rather just the max!!! This isn't what we want
    # first map from img1 to img2
    ans = [np.argmax(mat[i,:]) for i in range(mat.shape[0])]
    # then from img2 back to img1
    ans2 = [np.argmax(mat[:,i]) for i in ans]
    ans2 = np.array(ans2)
    perfect = list(range(mat.shape[0]))
    perfect = np.array(perfect)
    matches = ans2 == perfect
    # don't count background matching in any of the below
    n_matched = len(perfect[matches])-1
    n_gt = mat.shape[0]-1
    n_predict = mat.shape[1]-1
    print("{} Best Matches out of {} GT cells and {} predicted cells...".format(n_matched, n_gt, n_predict))
    # print("Perfect matches are: ", perfect[matches])
    return n_matched, n_gt, n_predict














