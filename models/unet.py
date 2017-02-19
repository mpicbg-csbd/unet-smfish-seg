doc="""
UNET architecture for pixelwise classification
"""

import numpy as np
import skimage.io as io

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.utils import np_utils
# from keras.utils.visualize_util import plot
from keras.optimizers import SGD

import skimage.util as ut

# import sys
# sys.path.insert(0, './lib/')
# from help_functions import *

# input image dimensions

def sample_patches_from_img(coords, img, x_width=48, y_width=48):
    assert coords[:,0].max() <= img.shape[0]-x_width
    assert coords[:,1].max() <= img.shape[1]-y_width
    patches = np.zeros(shape=(coords.shape[0], x_width, y_width), dtype=img.dtype)
    for m,ind in enumerate(coords):
        patches[m] = img[ind[0]:ind[0]+x_width, ind[1]:ind[1]+y_width]
    return patches

def random_patch_coords(img, n, x_width=48, y_width=48):
    xc = np.random.randint(img.shape[0]-x_width, size=n)
    yc = np.random.randint(img.shape[1]-y_width, size=n)
    return np.stack((xc, yc), axis=1)

def regular_patch_coords(img, x_width=48, y_width=48, step=10):
    dx,dy = img.shape[0]-x_width, img.shape[1]-y_width
    coords = ut.view_as_windows(np.indices((dx+1,dy+1)), (2,1,1), step=step)
    a,b,c,d,e,f = coords.shape
    return coords.reshape(b*c,2)

def rebuild_img_from_patches(zeros_img, patches, coords):
    "TODO: potentially add more ways of recombining than a simple average, i.e. maximum"
    count_img = np.zeros_like(zeros_img)
    n_samp, dx, dy = patches.shape
    for cord,patch in zip(coords, patches):
        x,y = cord
        zeros_img[x:x+dx, y:y+dy] += patch
        count_img[x:x+dx, y:y+dy] += np.ones_like(patch)
    return zeros_img/count_img

def imglists_to_XY(greylist, labellist, x_width=48, y_width=48, step=10):
    "turn list of images into ndarray of patches, labels and their coordinates"

    def get_patch_coords(img):
        coords = regular_patch_coords(img, x_width, y_width, step)
        return coords
        
    def f((img, coords)):
        return sample_patches_from_img(coords, img, x_width, y_width)

    coords = map(get_patch_coords, greylist)
    greypatches = map(f, zip(greylist, coords))
    X = np.concatenate(tuple(greypatches), axis=0)
    labelpatches = map(f, zip(labellist, coords))
    labelpatches = np.concatenate(tuple(labelpatches), axis=0)
    Y = labelpatches
    coords = np.concatenate(tuple(coords), axis=0)

    # reshape into theano dimension ordering
    a,b,c = X.shape
    assert K.image_dim_ordering() == 'th'
    X = X.reshape(a, 1, b, c)
    Y = Y.reshape(a, 1, b, c)

    # convert label values to vector of label scores
    Y[Y==2] = 1
    assert Y.min() == 0
    nb_classes = Y.max()+1
    Y = Y.reshape(a*b*c)
    Y = np_utils.to_categorical(Y, nb_classes)
    Y = Y.reshape(a, b*c, nb_classes)
    #Y = Y.reshape(a ,b,c, nb_classes).transpose(0,3,1,2)

    # normalize X per patch
    mi = np.amin(X,axis = (1,2,3), keepdims = True)
    ma = np.amax(X,axis = (1,2,3), keepdims = True)+1.e-10
    X = (X-mi)/(ma-mi)

    return X.astype(np.float32), Y.astype(np.float32), coords

def my_categorical_crossentropy(weights =(1., 1.)):
    def _func(y_true, y_pred):
        return -(weights[0]*K.mean(y_true[:,:,0]*K.log(y_pred[:,:,0]+K.epsilon()))+weights[1]*K.mean(y_true[:,:,1]*K.log(y_pred[:,:,1]+K.epsilon())))

        # return -(K.mean(y_true[:,:,0]*K.log(y_pred[:,:,0]+K.epsilon()))+K.mean(y_true[:,:,1]*K.log(y_pred[:,:,1]+K.epsilon())))
    return _func

def get_unet_small(patch_height, patch_width, n_ch):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv2), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    # here nb_classes used to be just the number 2
    conv6 = Convolution2D(2, 1, 1, activation='relu',border_mode='same')(conv5)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='sgd',loss=my_categorical_crossentropy((1,30.)),metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr = 0.0001),loss=my_categorical_crossentropy((1,1.)),metrics=['mean_squared_error'])
    model.compile(optimizer=Adam(lr = 0.001),loss="categorical_crossentropy", metrics=['accuracy'])

    return model

def get_unet(patch_height, patch_width, n_ch):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    # here nb_classes used to be just the number 2
    conv6 = Convolution2D(2, 1, 1, activation='relu',border_mode='same')(conv5)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer='sgd',loss=my_categorical_crossentropy((1,30.)),metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr = 0.001),loss=my_categorical_crossentropy((1,10.)),metrics=['mean_squared_error'])
    model.compile(optimizer=Adam(lr = 0.001),loss="categorical_crossentropy", metrics=['accuracy'])
    return model

def trainmodel(model, X_train, Y_train, X_vali, Y_vali, batch_size = 128, nb_epoch = 1, patience = 5):
    """
    Note: Input X,Y should really just be training data! Not all the labeled data we have!
    """
    import sklearn.utils as ut
    import util
    # from sklearn.model_selection import train_test_split

    path_model = "./keras_model.h5"

    classimg = np.argmax(Y_train, axis=-1).flatten()

    classweights = ut.compute_class_weight('balanced', list(np.unique(classimg)), classimg)
    checkpointer = ModelCheckpoint(filepath=path_model, verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(patience=patience, verbose=0)
    callbacks = [checkpointer, earlystopper]
    classweights = {i:classweights[i] for i in range(len(classweights))}
    # print(classweights)
    # model = get_unet(48, 48, 1)
    # model = buildmodel()
    # model.compile(loss='categorical_crossentropy',
    #             #   optimizer='adadelta',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_vali, Y_vali),
              #class_weight=classweights,
              callbacks=callbacks)
    # model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #           verbose=1, validation_data=(X_vali, Y_vali), sample_weight=np.array(sampleweights), callbacks=callbacks)

    score = model.evaluate(X_vali, Y_vali, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model

def view_results(X,Y,Ypred):
    import matplotlib.pyplot as plt
    def imsho(x, fname):
        # plt.figure()
        # plt.imshow(x, interpolation='nearest')
        io.imsave(fname, x)
    idx = np.random.randint(Ypred.shape[0])
    x= X[idx,0]
    a,b = x.shape
    # imsho(x, 'x.tif')
    y_gt = Y[idx,:,0].reshape((a,b))
    # imsho(y_gt)
    y_pre = Ypred[idx,:,0].reshape((a,b))
    # imsho(y_pre)
    io.imsave('randstack.tif', np.stack((x,y_gt,y_pre), axis=0))
