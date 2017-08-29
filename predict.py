import sys
sys.path.insert(0, "../.local/lib/python3.5/site-packages/")

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt

import util
import json
import numpy as np
import skimage.io as io
import unet
import datasets
import patchmaker
import train

rationale = """
Test out predict.py refactor.
"""

predict_params = {
 'savedir' : './',
 'grey_tif_folder' : None,
 'batch_size' : 1,
 'width': 1024,
 # 'full_imgs': "data3/labeled_data_membranes/images_big/smaller2x/",
}

def get_model_params_from_dir(predict_params, direc):
    pp = predict_params
    train_params = json.load(open(direc + '/train_params.json'))
    for key in ['n_convolutions_first_layer', 'n_pool', 'n_classes', 'dropout_fraction', 'itd', 'stakk', 'n_patches', 'split']:
        pp[key] = train_params.get(key, 'MISSING')
    mpgrid = 2**pp['n_pool']
    m,rm = divmod(pp['itd'], mpgrid)
    border = (m+1)*mpgrid   # at least as big as itd.
    pp['step'] = pp['width']-2*border
    pp['initial_model_params'] = direc + '/unet_model_weights_checkpoint.h5'
    return pp

def predict_all(predict_parms, data=None, history=None):
    """
    history is modified in place!
    full images is either None or the name of a folder containing greyscale images.
    """
    pp = predict_params
    model = train.get_model(predict_parms)
    print(model.summary())

    if data:
        X_train, X_vali, Y_train, Y_vali = data

    ## MAKE PRETTY PREDICTIONS
    if pp['grey_tif_folder']:
        full_image_names = util.sglob(pp['grey_tif_folder'] + '*.tif')
        for name in full_image_names[:5]:
            img = io.imread(name)
            print(name, img.shape)
            res = predict_single_image(model, img, pp)
            print("Res shape", res.shape)
            combo = np.stack((img, res), axis=0)
            path, base, ext =  util.path_base_ext(name)
            io.imsave(pp['savedir'] + "/" + base + '_predict_' + ext, combo.astype('float32'))

    def get_predictions_and_scores(X, Y):
        ypred = model.predict(unet.add_singleton_dim(X), pp['batch_size'])
        acc = accuracy(Y, ypred)
        ce  = crossentropy(Y, ypred)
        return ypred, (acc, ce)

    def plot_and_save(X,Y,fig, name):
        ypred, (acc, ce) = get_predictions_and_scores(X, Y)
        acc_ids = np.argsort(acc)
        ce_ids  = np.argsort(ce)
        fig.gca().plot(acc[acc_ids], '.', label='acc_'+name)
        fig.gca().plot(ce[ce_ids], '.', label='ce_'+name)

        ypred = ((2**16-1)*ypred[...,1]).astype('uint16')
        stakk = np.stack([X, Y, ypred], axis=1)
        stakk = stakk[ce_ids]
        io.imsave(pp['savedir'] + '/ypred_{}.tif'.format(name), stakk)

        if history:
            # in place
            history.history['ce_'+name]  = ce[ce_ids].tolist()
            history.history['acc_'+name] = acc[acc_ids].tolist()

    fig = plt.figure()
    if data:
        plot_and_save(X_train, Y_train, fig, 'train')
        plot_and_save(X_vali, Y_vali, fig, 'vali')
    X,_,Y,_ = train.build_XY(pp, n_patches=-1, split='noval')
    plot_and_save(X, Y, fig, 'all')
    plt.legend()
    plt.savefig(pp['savedir'] + '/acc_ce_dist.pdf')

def predict_single_image(model, img, pp):
    "unet predict on a greyscale img"

    coords = patchmaker.square_grid_coords(img, pp['step'])
    w = pp['width']
    X = patchmaker.sample_patches_from_img(coords, img, (w,w))
    # a,b = img.shape
    # n = pp['n_pool']
    # am = a%(2**n)
    # bm = b%(2**n)
    # X = img[np.newaxis, :-am, :-bm]
    X = unet.normalize_X(X)
    X = unet.add_singleton_dim(X)
    Y_pred = model.predict(X, batch_size=pp['batch_size'])

    if Y_pred.ndim == 3:
        print("NDIM 3, ")
        Y_pred = Y_pred.reshape((-1, pp['width'], pp['width'], pp['n_classes']))

    Y_pred = Y_pred[...,1]
    #Y_new = np.zeros((a,b))
    #Y_new[:-am, :-bm] = Y_pred
    #print("Y_pred shape: ", Y_pred.shape)
    #io.imsave('Ypred.tif', Y_pred)

    res = patchmaker.piece_together(Y_pred, coords, imgshape=img.shape, border=pp['itd'])
    return res[...,0].astype(np.float32)

def accuracy(ytrue, ypred):
    """compute accuracy, assume ytrue is labels, and ypred is dist-over-labels with an extra dim."""
    ypred_2 = np.argmax(ypred, axis=-1)
    masks = ytrue != ypred_2
    acc = np.sum(masks, axis=(1,2))/np.prod(masks[0].shape)
    return acc

def crossentropy(ytrue, ypred):
    # compute categorical crossentropy (unweighted)
    a,b,c = ytrue.shape
    ytrue = unet.np_utils.to_categorical(ytrue)
    ytrue = ytrue.reshape(a,b,c,2)
    ce = ytrue * np.log(ypred + 1.0e-7)
    ce = np.sum(ce, axis=3)
    ce = -np.mean(ce, axis=(1,2))
    return ce

if __name__ == '__main__':
    predict_params = get_model_params_from_dir(predict_params, sys.argv[1])
    predict_params['n_patches'] = 120
    predict_params['split'] = 6
    predict_params['savedir'] = sys.argv[2]
    predict_all(predict_params)

