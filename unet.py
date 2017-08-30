doc="""
UNET architecture for pixelwise classification
"""

import numpy as np
import skimage.io as io
import json

from keras.activations import softmax
from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import Input, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import warping
import patchmaker
import datasets

def normalize_X(X):
    # normalize min and max over X per patch to [0,1]
    X = X.astype('float32')
    #mi = np.amin(X,axis = (1,2), keepdims = True)
    #mi = np.percentile(X, 1, axis = (1,2), keepdims = True)
    #mi = np.percentile(X, 1)
    mi = X.min()
    X -= mi
    #ma = np.amax(X,axis = (1,2), keepdims = True) + 1.e-10
    #ma = np.percentile(X, 99, axis = (1,2), keepdims = True) + 1.e-10
    #ma = np.percentile(X, 99) + 1.e-10
    ma = X.max()
    X /= ma
    #X = np.clip(X, 0, 1)
    return X

def labels_to_activations(Y, n_classes=2):
    #assert Y.min() == 0
    a,b,c = Y.shape
    Y = Y.reshape(a*b*c)
    Y = np_utils.to_categorical(Y, n_classes)
    Y = Y.reshape(a, b, c, n_classes)
    return Y.astype(np.float32)

def add_singleton_dim(X):
    """
    backend [theano, tensorflow] dependent
    """
    a,b,c = X.shape
    if K.image_dim_ordering() == 'th':
        X = X.reshape((a, 1, b, c))
    elif K.image_dim_ordering() == 'tf':
        X = X.reshape((a, b, c, 1))
    return X
    
def my_categorical_crossentropy(weights=(1., 1.), itd=1, BEnd=K):
    """
    NOTE: The default weights assumes 2 classes, but the loss works for arbitrary classes if we simply change the length of the weights arg.
    
    Also, we can replace K with numpy to get a function we can actually evaluate (not just pass to compile)!
    """
    weights = np.array(weights)
    mean = BEnd.mean
    log  = BEnd.log
    summ = BEnd.sum
    eps  = K.epsilon()
    def catcross(y_true, y_pred):
        ## only use the valid part of the result! as if we had only made valid convolutions
        yt = y_true[:,itd:-itd,itd:-itd,:]
        yp = y_pred[:,itd:-itd,itd:-itd,:]
        ## NOTE: mean and sum commute here; we're still commuting the avg cross-entropy per pixel.
        ce = yt * log(yp + eps)
        ce = mean(ce, axis=(0,1,2))
        result = weights * ce
        result = -summ(result)
        return result
    return catcross

def get_unet_n_pool(n_pool, n_classes=2, n_convolutions_first_layer=32, dropout_fraction=0.2):
    """
    The info travel distance is given by info_travel_dist(n_pool, 3)
    """

    if K.image_dim_ordering() == 'th':
      inputs = Input((1, None, None))
      concatax = 1
      chan = 'channels_first'
    elif K.image_dim_ordering() == 'tf':
      inputs = Input((None, None, 1))
      concatax = 3
      chan = 'channels_last'

    def Conv(w):
        return Conv2D(w, (3,3), padding='same', data_format=chan, activation='relu', kernel_initializer='he_normal')
    def Pool():
        return MaxPooling2D(pool_size=(2,2), data_format=chan)
    def Upsa():
        return UpSampling2D(size=(2,2), data_format=chan)
    
    d = dropout_fraction
    
    def cdcp(s, inpt):
        """
        Conv, Drop, Conv, Pool
        """
        conv = Conv(s)(inpt)
        drop = Dropout(d)(conv)
        conv = Conv(s)(drop)
        pool = Pool()(conv)
        return conv, pool

    def uacdc(s, inpt, skip):
        """
        Up, cAt, Conv, Drop, Conv
        """
        up   = Upsa()(inpt)
        cat  = Concatenate(axis=concatax)([up, skip])
        conv = Conv(s)(cat)
        drop = Dropout(d)(conv)
        conv = Conv(s)(drop)
        return conv

    # once we've defined the above terms, the entire unet just takes a few lines ;)

    # holds the output of convolutions on the contracting path
    conv_layers = []

    # the first conv comes from the inputs
    s = n_convolutions_first_layer
    conv, pool = cdcp(s, inputs)
    conv_layers.append(conv)

    # then the recursively describeable contracting part
    for _ in range(n_pool-1):
        s *= 2
        conv, pool = cdcp(s, pool)
        conv_layers.append(conv)

    # the flat bottom. no max pooling.
    s *= 2
    conv_bottom = Conv(s)(pool)
    conv_bottom = Dropout(d)(conv_bottom)
    conv_bottom = Conv(s)(conv_bottom)
    
    # now each time we cut s in half and build the next UCCDC
    s = s//2
    up = uacdc(s, conv_bottom, conv_layers[-1])

    # recursively describeable expanding path
    for conv in reversed(conv_layers[:-1]):
        s = s//2
        up = uacdc(s, up, conv)

    # final (1,1) convolutions and activation
    acti_layer = Conv2D(n_classes, (1, 1), padding='same', data_format=chan, activation=None)(up)
    if K.image_dim_ordering() == 'th':
        acti_layer = core.Permute((2,3,1))(acti_layer)
    acti_layer = core.Activation(softmax)(acti_layer)
    model = Model(inputs=inputs, outputs=acti_layer)
    return model

def info_travel_dist(n_maxpool, conv=3):
    """
    n_maxpool: number of 2x downsampling steps
    conv: the width of the convolution kernel (e.g. "3" for standard 3x3 kernel.)
    returns: the info travel distance == the amount of width that is lost in a patch / 2
    """
    conv2 = 2*(conv-1)
    width = 0
    for i in range(n_maxpool):
        width -= conv2
        width /= 2
    width -= conv2
    for i in range(n_maxpool):
        width *= 2
        width -= conv2
    return int(-width/2)

# ---- PUBLIC INTERFACE ----

def train_unet(X_train, Y_train, X_vali, Y_vali, model, train_params):
    tp = train_params

    print("COMPUTE CLASSWEIGHTS")
    _, counts = np.unique(Y_train, return_counts=True)
    weights = (1-counts/counts.sum())/(len(counts)-1)
    print("ClassWeights:", weights)

    print("SETUP OPTIMIZER")
    if tp['optimizer'] == 'sgd':
        optim = SGD(lr=tp['learning_rate'], momentum=tp['momentum'])
    elif tp['optimizer'] == 'adam':
        optim = Adam(lr = tp['learning_rate'])

    model.compile(optimizer=optim, loss=my_categorical_crossentropy(weights=weights, itd=tp['itd']), metrics=['accuracy'])

    print("SETUP CALLBACKS")
    checkpointer = ModelCheckpoint(filepath=tp['savedir'] + "/unet_model_weights_checkpoint.h5", verbose=0, save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(patience=tp['patience'], verbose=0)
    callbacks = [checkpointer, earlystopper]

    history = model.fit_generator(
                batch_generator_patches(X_train, Y_train, train_params),
                steps_per_epoch=tp['batches_per_epoch'],
                epochs=tp['epochs'],
                verbose=1,
                callbacks=callbacks,
                validation_data=(add_singleton_dim(X_vali), labels_to_activations(Y_vali, tp['n_classes'])))

    print("FINISHED TRAINING")

    history.history['X_train_shape'] = X_train.shape
    history.history['X_vali_shape'] = X_vali.shape

    if False:
        Y_pred_train = model.predict(add_singleton_dim(X_train), tp['batch_size'])
        Y_pred_vali  = model.predict(add_singleton_dim(X_vali), tp['batch_size'])

        print("ALL THE SHAPES")
        print(X_train.shape, Y_train.shape, Y_pred_train.shape)
        print(X_vali.shape, Y_vali.shape, Y_pred_vali.shape)

        def savetiff(fname, img):
            io.imsave(fname, img, plugin='tifffile', compress=6)

        if Y_pred_train.ndim == 3:
            print("NDIM 3, ")
            Y_pred_train = Y_pred_train.reshape((-1, y_width, x_width, tp['n_classes']))
            Y_pred_vali  = Y_pred_vali.reshape((-1, y_width, x_width, tp['n_classes']))

        def toUint16(X):
            X -= X.min()
            X *= (2**16-1)/X.max()
            X = X.astype('uint16')
            return X

        X_train, X_vali = toUint16(X_train), toUint16(X_vali)
        
        res = np.stack((X_train, Y_train, np.argmax(Y_pred_train, axis=-1)), axis=1)
        savetiff(tp['savedir'] + '/training.tif', res.astype('uint16'))
        res = np.stack((X_vali, Y_vali, np.argmax(Y_pred_vali, axis=-1)), axis=1)
        savetiff(tp['savedir'] + '/testing.tif', res.astype('uint16'))

    return history

def batch_generator_patches(X, Y, train_params, verbose=False):
    epoch = 0
    tp = train_params
    while (True):
        epoch += 1
        current_idx = 0
        batchnum = 0
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds]
        Y = Y[inds]
        while batchnum < tp['batches_per_epoch']:
            Xbatch, Ybatch = X[current_idx:current_idx + tp['batch_size']].copy(), Y[current_idx:current_idx + tp['batch_size']].copy()
            # io.imsave('X.tif', Xbatch, plugin='tifffile')
            # io.imsave('Y.tif', Ybatch, plugin='tifffile')

            current_idx += tp['batch_size']

            for i in range(Xbatch.shape[0]):
                x = Xbatch[i]
                y = Ybatch[i]
                x,y = warping.randomly_augment_patches(x, y, tp['noise'], tp['flipLR'], tp['warping_size'], tp['rotate_angle_max'])
                Xbatch[i] = x
                Xbatch = normalize_X(Xbatch)
                Ybatch[i] = y

            # io.imsave('Xauged.tif', Xbatch.astype('float32'), plugin='tifffile')
            # io.imsave('Yauged.tif', Ybatch.astype('float32'), plugin='tifffile')

            Xbatch = add_singleton_dim(Xbatch)
            Ybatch = labels_to_activations(Ybatch, tp['n_classes'])

            # print('xshape', Xbatch.shape)
            # print('yshape', Ybatch.shape)

            batchnum += 1
            yield Xbatch, Ybatch

