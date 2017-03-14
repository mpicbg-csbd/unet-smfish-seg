


def batch_generator_patches(indir,
                            n_batch=128,
                            shuffle=True,
                            verbose=False,
                            prefix="*.npz"):
    """
    generator that returns n_batch patches from all the npz files in indir infinetly 
    (each containing the field "X" and "Y")
    returns X,Y
    """
    files = sglob(os.path.join(os.path.expanduser(indir), prefix))

    if verbose:
        print(files)

    if len(files) == 0:
        raise ValueError("no files found in %s" % indir)

    files.sort()

    X_acc, Y_acc = None, None

    count = 0
    while (True):
        if shuffle:
            fname = files[np.random.randint(0, len(files))]
        else:
            fname = files[count]

        count = (count + 1) % len(files)

        if verbose:
            print("loading from %s ..." % fname)

        # f = np.load(fname)
        f = io.imread(fname)

        X, Y = f["X"], f["Y"]

        if verbose:
            print("found %s patches ..." % len(X))

        if X_acc is None:
            X_acc = X.copy()
            Y_acc = Y.copy()
        else:
            X_acc = np.concatenate([X_acc, X], axis = 0)
            Y_acc = np.concatenate([Y_acc, Y], axis = 0)

        offset = 0
        while offset+n_batch <= len(X_acc):
            if verbose:
                print("yielding")

            yield X_acc[offset:offset+n_batch].copy(), Y_acc[offset:offset+n_batch].copy()
            offset += n_batch
        X_acc, Y_acc = X_acc[offset:].copy(), Y_acc[offset:].copy()


def upscale_and_compare(labeling, annotated):
    a,b = labeling.shape
    _,c,d = annotated.shape
    upscaled = zoom(labeling, (c/a, d/b), order=0)
    score = label_imgs.match_score_1(annotated[0], upscaled)
    imsave('upscaled.tif', upscaled)
    imsave('cells.tif', annotated[0])
    return score

def compare_segment_predictions_with_groundtruth(segs, labels):
    "segs and labels are lists of filenames of images."
    from label_imgs import match_score_1
    def print_and_score(s_l):
        s,l = s_l
        simg = imread(s)
        limg = imread(l)
        print('\n', s)
        return match_score_1(simg, limg)
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
    imgs = [imread(mem) for mem in membranes]
    res = [get_label(img, threshold) for img in imgs]
    for fname, img in zip(membranes, res):
        path, base, ext = util.path_base_ext(fname)
        imsave(base + '_seg' + ext, img) 
        imsave(base + '_seg_preview' + ext, label_imgs.labelImg_to_rgb(img))
    return res
