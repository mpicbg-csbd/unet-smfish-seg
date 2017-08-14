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
# from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator

import skimage.util as skut
import util
import warping
import patchmaker
import datasets

# example param values. Set them in train.py
nb_classes = 2

def labels_to_activations(Y):
    assert Y.min() == 0
    a,b,c = Y.shape
    Y = Y.reshape(a*b*c)
    Y = np_utils.to_categorical(Y, nb_classes)
    # Y = Y.reshape(a, b*c, nb_classes)
    Y = Y.reshape(a, b, c, nb_classes)
    return Y.astype(np.float32)

def add_singleton_dim(X):
    # reshape into theano/tensorflow dimension ordering
    a,b,c = X.shape
    if K.image_dim_ordering() == 'th':
        X = X.reshape((a, 1, b, c))
    elif K.image_dim_ordering() == 'tf':
        X = X.reshape((a, b, c, 1))
    return X

# setup and train the model

def my_categorical_crossentropy(weights=(1., 1.), itd=0):
    # replace K with numpy (and eps with 1e-7) to get a function we can actually evaluate (not just pass to compile)!
    mean = K.mean
    log = K.log
    eps = K.epsilon()
    def catcross(y_true, y_pred):
        # only use the valid part of the result! as if we had only made valid convolutions and done reshaping
        y_true_valid = y_true[:,itd:-itd,itd:-itd,:]
        y_pred_valid = y_pred[:,itd:-itd,itd:-itd,:]
        return -(weights[0] * mean(y_true_valid[:,:,:,0] * log(y_pred_valid[:,:,:,0] + eps)) +
                 weights[1] * mean(y_true_valid[:,:,:,1] * log(y_pred_valid[:,:,:,1] + eps)))
    return catcross

def get_unet_n_pool(n_pool, n_convolutions_first_layer=32, dropout_fraction=0.2):
    """
    The info travel distance is given by analysis.info_travel_dist(n_pool, 3)
    """

    if K.image_dim_ordering() == 'th':
      inputs = Input((1, None, None))
      concatax = 1
      chan = 'channels_first'
    if K.image_dim_ordering() == 'tf':
      inputs = Input((None, None, 1))
      concatax = 3
      chan = 'channels_last'

    def Conv(w):
        return Conv2D(w, (3,3), padding='same', data_format=chan, activation='relu')
    def Pool():
        return MaxPooling2D(pool_size=(2,2), data_format=chan)
    def Upsa():
        return UpSampling2D(size=(2,2), data_format=chan)
    
    def cdcp(s, inpt):
        """
        conv, drop, conv, pool
        """
        conv = Conv(s)(inpt)
        drop = Dropout(d)(conv)
        conv = Conv(s)(drop)
        pool = Pool()(conv)
        return conv, pool

    def uccdc(s, inpt, skip):
        """
        up, cat, conv, drop, conv
        """
        up   = Upsa()(inpt)
        cat  = Concatenate(axis=concatax)([up, skip])
        conv = Conv(s)(cat)
        drop = Dropout(d)(conv)
        conv = Conv(s)(drop)
        return conv

    # once we've defined the above terms, the entire unet just takes a few lines ;)
    s = n_convolutions_first_layer
    d = dropout_fraction
    
    # holds the output of convolutions on the contracting path
    conv_layers = []

    # the first conv comes from the inputs
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
    up = uccdc(s, conv_bottom, conv_layers[-1])

    # recursively describeable expanding path
    for conv in reversed(conv_layers[:-1]):
        s = s//2
        up = uccdc(s, up, conv)

    # final (1,1) convolutions and activation
    acti_layer = Conv2D(2, (1, 1), padding='same', data_format=chan, activation='relu')(up)
    if K.image_dim_ordering() == 'th':
        acti_layer = core.Permute((2,3,1))(acti_layer)
    acti_layer = core.Activation(softmax)(acti_layer)
    model = Model(inputs=inputs, outputs=acti_layer)
    return model

# ---- PUBLIC INTERFACE ----

def train_unet(X_train, Y_train, X_vali, Y_vali, model, train_params):

    tp = train_params

    print("SETUP THE CLASSWEIGHTS")
    # IMPORTANT! The weight for membrane is given by the fraction of non-membrane! (and vice versa)
    classimg = Y_train.flatten()
    non_zeros = len(classimg[classimg!=0])
    non_ones = len(classimg[classimg!=1]) # * membrane_weight_multiplier
    total = non_zeros + non_ones
    w0 = non_zeros / total
    w1 = non_ones / total
    class_relative_frequncies = {0: w0, 1: w1}
    print(class_relative_frequncies)

    print("SETUP OPTIMIZER")
    if tp['optimizer'] == 'sgd':
        optim = SGD(lr=tp['learning_rate'], momentum=tp['momentum'])
    elif tp['optimizer'] == 'adam':
        optim = Adam(lr = tp['learning_rate'])

    model.compile(optimizer=optim, loss=my_categorical_crossentropy(weights=(w0, w1), itd=tp['itd']), metrics=['accuracy'])

    # Setup callbacks
    print("SETUP CALLBACKS")
    checkpointer = ModelCheckpoint(filepath=tp['savedir'] + "/unet_model_weights_checkpoint.h5", verbose=0, save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(patience=tp['patience'], verbose=0)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks = [checkpointer, earlystopper]

    history = model.fit_generator(
                batch_generator_patches(X_train, Y_train, train_params),
                steps_per_epoch=tp['steps_per_epoch'],
                epochs=tp['epochs'],
                verbose=1,
                callbacks=callbacks,
                validation_data=(add_singleton_dim(X_vali), labels_to_activations(Y_vali)))

    print("FINISHED TRAINING")

    # history.history['steps_per_epoch'] = steps_per_epoch
    history.history['X_train_shape'] = X_train.shape
    history.history['X_vali_shape'] = X_vali.shape

    Y_pred_train = model.predict(add_singleton_dim(X_train), tp['batch_size'])
    Y_pred_vali = model.predict(add_singleton_dim(X_vali), tp['batch_size'])

    print("ALL THE SHAPES")
    print(X_train.shape, Y_train.shape, Y_pred_train.shape)
    print(X_vali.shape, Y_vali.shape, Y_pred_vali.shape)

    def savetiff(fname, img):
        io.imsave(fname, img, plugin='tifffile', compress=6)

    if Y_pred_train.ndim == 3:
        print("NDIM 3, ")
        Y_pred_train = Y_pred_train.reshape((-1, y_width, x_width, 2))
        Y_pred_vali = Y_pred_vali.reshape((-1, y_width, x_width, 2))

    res = np.stack((X_train, Y_train, Y_pred_train[...,1]), axis=2)
    savetiff(tp['savedir'] + '/training.tif', res)
    res = np.stack((X_vali, Y_vali, Y_pred_vali[...,1]), axis=2)
    savetiff(tp['savedir'] + '/testing.tif', res)

    return history

def batch_generator_patches(X,Y, train_params, verbose=False):
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
        while batchnum < tp['steps_per_epoch']:
            Xbatch, Ybatch = X[current_idx:current_idx + tp['batch_size']].copy(), Y[current_idx:current_idx + tp['batch_size']].copy()
            # io.imsave('X.tif', Xbatch, plugin='tifffile')
            # io.imsave('Y.tif', Ybatch, plugin='tifffile')

            current_idx += tp['batch_size']

            for i in range(Xbatch.shape[0]):
                x = Xbatch[i]
                y = Ybatch[i]
                x,y = warping.randomly_augment_patches(x, y, tp['noise'], tp['flipLR'], tp['warping_size'], tp['rotate_angle_max'])
                Xbatch[i] = x
                Ybatch[i] = y

            # io.imsave('Xauged.tif', Xbatch.astype('float32'), plugin='tifffile')
            # io.imsave('Yauged.tif', Ybatch.astype('float32'), plugin='tifffile')

            Xbatch = add_singleton_dim(Xbatch)
            Ybatch = labels_to_activations(Ybatch)

            batchnum += 1
            yield Xbatch, Ybatch

# use the model for prediction

def predict_single_image(model, img, itd, batch_size=32):
    "unet predict on a greyscale img"
    X = datasets.imglist_to_X([img])
    X2 = add_singleton_dim(X)
    Y_pred = model.predict(X2, batch_size=batch_size)

    if Y_pred.ndim == 3:
        print("NDIM 3, ")
        Y_pred = Y_pred.reshape((-1, y_width, x_width, 2))

    Y_pred = Y_pred[...,1]
    print("Y_pred shape: ", Y_pred.shape)
    io.imsave('Ypred.tif', Y_pred)
    # WARNING TODO: This will break when we change the coords used in `imglist_to_X`

    coords = patchmaker.square_grid_coords(img, step)
    res = patchmaker.piece_together(Y_pred, coords, imgshape=img.shape, border=itd)
    #xres = patchmaker.piece_together(X, coords, imgshape=img.shape, border=itd)
    print("TEST*: ", img.shape, res.shape)
    return res[...,0].astype(np.float32)
    #return xres[...,0].astype(np.float32)
