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
import analysis
import datasets

from scipy.ndimage import zoom
import patchmaker
import skimage.io as io

rationale = """
Find the BUG!
"""

train_params = {
 'savedir' : './',

 #'stakk'   : 'stakk_400_512_comp.tif',
 'stakk'   : 'stakk_smalltest.tif',

 'batch_size' : 1,
 'membrane_weight_multiplier' : 1,
 'epochs' : 100,
 'patience' : 30,
 'batches_per_epoch' : "TBD",

 'optimizer' : 'adam', # 'sgd' or 'adam' (adam ignores momentum)
 'learning_rate' : 1.00e-4, #3.16e-5,
 'momentum' : 0.99,

 ## noise True | False
 'noise':False,
 ## warping_size > 0, float
 'warping_size' : 0,
 ## flipLR True | False
 'flipLR' : True,
 ## rotate_angle_max > 0, float
 'rotate_angle_max' : 0,

 'initial_model_params' : None, #"training/m158/unet_model_weights_checkpoint.h5",
 'n_pool' : 2,
 'n_classes' : 2,
 'n_convolutions_first_layer' : 32,
 'dropout_fraction' : 0.2,
 'itd' : 10, # border ~= info_travel_dist(n_pool)
}

def fix_labels(Y):
    Y[Y!=0]=2
    Y[Y==0]=1
    Y[Y==2]=0
    return Y

def stakk_to_XY(train_params, split=7):
    """
    stakk -> X,Y train & vali
    """
    stakk = io.imread(train_params['stakk'])
    a,b,c,d = stakk.shape

    ## TODO: REMOVE THIS!!!! 
    # stakk = stakk[:a//2]

    xs = stakk[:,0,...]
    xs = xs.astype('float32')
    xs /= xs.max(axis=(1,2), keepdims=True)
    ys = fix_labels(stakk[:,1,...])
    xmask = xs.sum(axis=(1,2))>5500 # bright
    ymask = ys.sum(axis=(1,2))>0 # have membrane
    # xs = xs[ymask]
    # ys = ys[ymask]

    print('xmask.sum', xmask.sum())
    print('ymask.sum', ymask.sum())

    a,b,c = xs.shape
    end = a//split
    X_train = xs[:-end, ...]
    X_vali  = xs[-end:, ...]
    Y_train = ys[:-end, ...]
    Y_vali  = ys[-end:, ...]
    return X_train, X_vali, Y_train, Y_vali

def train(train_params):
    start_time = time.time()

    ## Now finalize and save the train params

    #train_params['itd'] = analysis.info_travel_dist(train_params['n_pool'])
    train_params['rationale'] = rationale

    # train162 = io.imread('training/m162/training.tif')
    # test162  = io.imread('training/m162/testing.tif')
    # X_train, Y_train = train162[:100,...,0], train162[:100,...,1].astype('uint16')
    # X_vali, Y_vali = test162[:100,...,0], test162[:100,...,1].astype('uint16')

    X_train, X_vali, Y_train, Y_vali = stakk_to_XY(train_params, split=3)

    train_params['batches_per_epoch'], _ = divmod(X_train.shape[0], train_params['batch_size'])
    json.dump(train_params, open(train_params['savedir'] + '/train_params.json', 'w'))

    print("SHAPES AND TYPES AND MINMAX.")
    print(X_train.shape, Y_train.shape)
    print(X_train.dtype, Y_train.dtype)
    print(X_train.min(), X_train.max())
    print(Y_train.min(), Y_train.max())
    print(X_vali.shape, Y_vali.shape)
    print(X_vali.dtype, Y_vali.dtype)
    print(X_vali.min(), X_vali.max())
    print(Y_vali.min(), Y_vali.max())
    print("Nans?:", np.isnan(X_train.flatten()).sum())

    ## build the model, maybe load pretrained weights.

    model = unet.get_unet_n_pool(train_params['n_pool'],
                                 n_classes = train_params['n_classes'],
                                 n_convolutions_first_layer = train_params['n_convolutions_first_layer'],
                                 dropout_fraction = train_params['dropout_fraction'])
    if train_params['initial_model_params']:
        model.load_weights(train_params['initial_model_params'])
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
    json.dump(history.history, open(train_params['savedir'] + '/history.json', 'w'))

    ## MAKE PRETTY PREDICTIONS
    import predict
    pp = predict.predict_params
    pp = predict.get_params_from_dir(pp, train_params['savedir'])
    pp['savedir'] = train_params['savedir']
    predict.predict(pp, model=model)

    return history

if __name__ == '__main__':
    train_params['savedir'] = sys.argv[1]
    train(train_params)
