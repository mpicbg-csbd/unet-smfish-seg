import sys
sys.path.insert(0, "/projects/project-broaddus/.local/lib/python3.5/site-packages/")
import pkg_resources
pkg_resources.require("scikit-image>=0.13.0")

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unet
import util
import time
import json
import numpy as np
import datasets
import matplotlib.pyplot as plt
import predict

from scipy.ndimage import zoom
import patchmaker
import skimage.io as io

rationale = """
Test refactor of predictions.
"""

train_params = {
 'savedir' : './',

 #'stakk' : 'stakk_400_512_comp.tif',
 #'stakk' : 'stakk_smalltest.tif',
 #'stakk' : 'stakk_all_256_128.tif',
 #'stakk' : 'stakk_800_1024_comp.tif',
 #'stakk' : 'stakk_all_296_480.tif',
 'stakk'  : 'stakk_mem_2xdown_256.tif',
 'n_patches' : 60,
 'split'  : 6,

 'batch_size' : 1,
 'membrane_weight_multiplier' : 1,
 'epochs' : 9,
 'patience' : 30,
 'batches_per_epoch' : "TBD",

 'optimizer' : 'adam', # 'sgd' or 'adam' (adam ignores momentum)
 'learning_rate' : 1.00e-4, #3.16e-5,
 'momentum' : 0.99,

 ## noise True | False
 'noise' : True,
 ## warping_size > 0, float
 'warping_size' : 0,
 ## flipLR True | False
 'flipLR' : True,
 ## rotate_angle_max > 0, float, rotations in [-angle, angle]
 'rotate_angle_max' : np.pi,

 'initial_model_params' : "training/m253/unet_model_weights_checkpoint.h5",
 'n_pool' : 2,
 'n_classes' : 2,
 'n_convolutions_first_layer' : 64,
 'dropout_fraction' : 0.20,
 'itd' : 24,
}

def fix_labels(Y):
    #Y[Y!=0]=2
    #Y[Y==0]=1
    Y[Y==2]=1
    return Y

def learn_background(Y):
    Y[Y!=4]=1
    Y[Y==4]=0
    return Y

def build_XY(train_params, n_patches=-1, split=7):
    """
    stakk -> X,Y train & vali
    """
    #tp = train_params
    stakk = io.imread(train_params['stakk'])

    ## Only train on a fraction of data
    if n_patches==-1:
        n_patches=stakk.shape[0]
    # stakk = stakk[[9,10]]

    ## Load and prepare
    xs = stakk[:,0]
    ys = stakk[:,1]

    xs = xs.astype('float32')
    xs = unet.normalize_X(xs)
    ys = fix_labels(ys)
    #ys = learn_background(ys)

    ## select data by characteristics
    # xmask = xs.mean(axis=(1,2))>0.6 # bright
    y_mem = ys.sum(axis=(1,2)) # have membrane
    y_mem_sort = np.argsort(y_mem)
    xs = xs[y_mem_sort][-n_patches:]
    ys = ys[y_mem_sort][-n_patches:]

    ## take random subset
    # inds = np.arange(stakk.shape[0])
    # np.shuffle(inds)
    # inds = inds[:n_patches]
    # xs = xs[inds]
    # ys = ys[inds]

    ## shuffle
    inds = np.arange(xs.shape[0])
    np.random.shuffle(inds)
    xs = xs[inds]
    ys = ys[inds]

    ## split into training and validation
    if split=='copy':
        X_train = xs
        X_vali  = xs.copy()
        Y_train = ys
        Y_vali  = ys.copy()
    elif split=='noval':
        X_train = xs
        X_vali  = None
        Y_train = ys
        Y_vali  = None
    elif type(split) is int:
        end = xs.shape[0]//split
        X_train = xs[:-end, ...]
        X_vali  = xs[-end:, ...]
        Y_train = ys[:-end, ...]
        Y_vali  = ys[-end:, ...]
    
    return X_train, X_vali, Y_train, Y_vali

def get_model(train_params):
    model = unet.get_unet_n_pool(train_params['n_pool'],
                             n_classes = train_params['n_classes'],
                             n_convolutions_first_layer = train_params['n_convolutions_first_layer'],
                             dropout_fraction = train_params['dropout_fraction'])
    if train_params['initial_model_params']:
        model.load_weights(train_params['initial_model_params'])
    return model

def train(train_params):
    start_time = time.time()

    ## Now finalize and save the train params

    train_params['rationale'] = rationale

    X_train, X_vali, Y_train, Y_vali = build_XY(train_params, n_patches=train_params['n_patches'], split=train_params['split'])

    train_params['batches_per_epoch'], _ = divmod(X_train.shape[0], train_params['batch_size'])
    json.dump(train_params, open(train_params['savedir'] + '/train_params.json', 'w'))

    def print_description(X,Y):
        print("SHAPES AND TYPES AND MINMAX.")
        print(X.shape, Y.shape)
        print(X.dtype, Y.dtype)
        print(X.min(), X.max())
        print(Y.min(), Y.max())
        print("Nans?:", np.isnan(X.flatten()).sum())

    print_description(X_train, Y_train)
    print_description(X_vali, Y_vali)

    ## build the model, maybe load pretrained weights.

    model = get_model(train_params)
    print(model.summary())

    ## MAGIC HAPPENS HERE
    begin_training_time = time.time()
    history = unet.train_unet(X_train, Y_train, X_vali, Y_vali, model, train_params)
    finished_time = time.time()

    ## MAGIC FINISHED, NOW SAVE TIMINGS
    history.history['warm_up_time'] = begin_training_time - start_time
    train_time = finished_time - begin_training_time
    history.history['train_time'] = train_time
    trained_epochs = len(history.history['acc'])
    history.history['trained_epochs'] = trained_epochs
    history.history['avg_time_per_epoch'] = train_time / trained_epochs
    history.history['avg_time_per_batch'] = train_time / (trained_epochs * train_params['batches_per_epoch'])
    history.history['avg_time_per_sample'] = train_time / (trained_epochs * history.history['X_train_shape'][0])

    print(history.history)
    json.dump(history.history, open(train_params['savedir'] + '/history.json', 'w'))

    data = X_train, X_vali, Y_train, Y_vali
    #train_params['initial_model_params'] = train_params['savedir'] + "/unet_model_weights_checkpoint.h5"
    pp = predict.predict_params
    pp['savedir'] = train_params['savedir']
    pp = predict.get_model_params_from_dir(pp, train_params['savedir'])
    print(pp)
    predict.predict_all(pp, data, history, full_imgs='data3/labeled_data_membranes/images_big/smaller2x/')
    json.dump(history.history, open(train_params['savedir'] + '/history.json', 'w'))
    return model, history

if __name__ == '__main__':
    train_params['savedir'] = sys.argv[1]
    train(train_params)
