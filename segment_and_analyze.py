import skimage.io as io
import numpy as np
from scipy.ndimage import label
import label_imgs
import util

import matplotlib.pyplot as plt
plt.ion()

def upscale_and_compare(labeling, annotated):
    a,b = labeling.shape
    _,c,d = annotated.shape
    upscaled = zoom(labeling, (c/a, d/b), order=0)
    score = label_imgs.match_score_1(annotated[0], upscaled)
    io.imsave('upscaled.tif', upscaled)
    io.imsave('cells.tif', annotated[0])
    return score

def compare_segment_predictions_with_groundtruth(segs, labels):
    "segs and labels are lists of filenames of images."
    def print_and_score(s_l):
        s,l = s_l
        simg = io.imread(s)
        limg = io.imread(l)
        print('\n', s)
        return label_imgs.match_score_1(simg, limg)
    return map(print_and_score, zip(segs, labels))

def get_label(img, threshold):
    "normalizes img min&max to [0,1), then binarize at threshold, then labels connected components."
    img = img.astype(np.float32, copy = False)
    img = np.nan_to_num(img) # sets nan to zero?
    img /= img.max()

    # threshold = threshold_otsu(img)

    # x = (1-threshold) * 0.22
    # threshold += x

    # img < threshold means the membrane takes on high values and we want the cytoplasm
    mask = np.where(img < threshold, 1, 0)

    lab_img = label(mask)[0]
    print("Number of cells: ", lab_img.max())

    # convert from int32
    lab_img = np.array(lab_img, dtype='uint16')
    return lab_img

def segment_classified_images(membranes, threshold):
    "membranes is a list of filenames of membrane images."
    imgs = [io.imread(mem) for mem in membranes]
    res = [get_label(img, threshold) for img in imgs]
    for fname, img in zip(membranes, res):
        path, base, ext = util.path_base_ext(fname)
        io.imsave(base + '_seg' + ext, img) 
        io.imsave(base + '_seg_preview' + ext, label_imgs.labelImg_to_rgb(img))
    return res

def combine_into_readme_img(greys, mems, segs):
    """
    all arguments are lists of .tif file names.
    segs should be the colorized, Preview versions.
    """
    grey_imgs = [io.imread(img) for img in greys]
    mem_imgs  = [io.imread(img) for img in mems]
    seg_imgs  = [io.imread(img) for img in segs]
    height_min = min([img.shape[0] for img in grey_imgs])
    width_min  = min([img.shape[1] for img in grey_imgs])
    height_max = max([img.shape[0] for img in grey_imgs])
    width_max  = max([img.shape[1] for img in grey_imgs])
    
    # print([img.shape for img in (grey_imgs[0], mem_imgs[0], seg_imgs[0])])
    
    for i in range(len(grey_imgs)):
        g = grey_imgs[i]
        m = mem_imgs[i]
        s3 = seg_imgs[i]

        g3 = np.stack((g,g,g), axis=-1)
        m3 = np.stack((m,m,m), axis=-1)

        plt.figure()
        plt.imshow(g3)
        plt.figure()
        plt.imshow(m3)
        plt.figure()
        plt.imshow(s3)

        res = np.concatenate((g3,m3,s3), axis=0) # height axis
        io.imsave('readme_imgs/readme_img_{:02d}.png'.format(i), res)