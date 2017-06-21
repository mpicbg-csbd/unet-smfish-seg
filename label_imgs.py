

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

# img2 = io.imread("./20150513_New_data/20150430_eif4g_dome03_slice11.tif")
# io.imsave('i1.tif', img1[1])

# img1 = io.imread("data/Cell_segmentations_paper/20150430_eif4g_dome03_R3D_MASKS.tif")
# img1 = img1[0]
# img2 = img1.copy()
# img2 = np.roll(img2, 3, axis=0)
# img2 = np.roll(img2, 3, axis=1)

# # this adds an edge whenever there is *any* overlap
# dg = nx.DiGraph()
# for i,j in zip(img1.flatten(), img2.flatten()):
#     dg.add_edge(i,j)

# first build graph matrix
# from scipy.ndimage import zoom
# img1 = zoom(img1, 0.2)
# img2 = zoom(img2, 0.2)


import colorsys

def label_colors(bg_ID=1, membrane_ID=0, n_colors = 10, maxlabel=1000):
    n_colors = 10
    HSV_tuples = [(x * 1.0 / n_colors, 0.5, 0.5) for x in range(n_colors)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    # intens *= 2**16/intens.max()

    assert membrane_ID != bg_ID
    RGB_tuples *= maxlabel
    RGB_tuples[membrane_ID] = (0, 0, 0)
    RGB_tuples[bg_ID] = (0.01, 0.01, 0.01)
    return RGB_tuples

def labelImg_to_rgb(img, bg_ID=1, membrane_ID=0):
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

def seg(ground_truth, prediction):
    """
    ground_truth and prediction are label-images (2d ndarrays).
    See definition of SEG: http://ctc2015.gryf.fi.muni.cz/Public/Documents/SEG.pdf
    """

    mat = matching_matrix(ground_truth, prediction)

    def jaccard(i):
        gt_size = np.sum(mat[i,:]) # an int
        max_match_ind = np.argmax(mat[i,:])
        if max_match_ind==0:
            print("match to background: ", i)
            return 0
        intersection = mat[i, max_match_ind]
        if 2*intersection > gt_size:
            # we have a good match!
            match_size = np.sum(mat[:,max_match_ind])
            print(gt_size, intersection, match_size, "matched to id:", max_match_ind)
            jac = intersection / (gt_size + match_size - intersection)
            print("jac:", jac)
            return jac
        else:
            return 0
    return [jaccard(i) for i in range(1, mat.shape[0])]

def matching_matrix(img1, img2):
    "img1 and img2 must be same shape, and label (uint) images."
    imgs = np.stack((img1, img2), axis=2)
    mat = np.zeros((img1.max()+1, img2.max()+1), dtype=np.uint32)
    a,b,c = imgs.shape
    for edg in imgs.reshape((a*b,c)):
        mat[edg[0], edg[1]] += 1
    return mat

def match_score_1(img1, img2):
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
    mat = matching_matrix(img1, img2)
    # TODO: not 50%, but rather just the max!!! This isn't what we want
    # first map from img1 to img2
    ans = [np.argmax(mat[i,:]) for i in range(mat.shape[0])]
    # then from img2 back to img1
    ans2 = [np.argmax(mat[:,i]) for i in ans]
    ans2 = np.array(ans2)
    perfect = list(range(mat.shape[0]))
    perfect = np.array(perfect)
    matches = ans2 == perfect
    n_matched = len(perfect[matches])
    n_gt = mat.shape[0]-1
    n_predict = mat.shape[1]-1
    print("{} Best Matches out of {} GT cells and {} predicted cells...".format(n_matched, n_gt, n_predict))
    # print("Perfect matches are: ", perfect[matches])
    return n_matched, n_gt, n_predict



# or weighted edges with weight corresponding to the number of overlapping pixels















