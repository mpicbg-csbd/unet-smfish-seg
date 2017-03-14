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

import skimage.util as skut
import util

# global variables for patch dimensions and stride
x_width = 160
y_width = 160
step = 10


# setup X and Y for feeding into the model

def sample_patches(data, patch_size, n_samples=100, verbose=False):
    """
    sample 2d patches of size patch_size from data
    """

    assert np.all([s <= d for d, s in zip(data.shape, patch_size)])

    # change filter_mask to something different if needed
    filter_mask = np.ones_like(data)

    # get the valid indices
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, data.shape)])
    valid_inds = np.where(filter_mask[border_slices])

    if len(valid_inds[0]) == 0:
        raise Exception("could not find anything to sample from...")

    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]

    # sample
    sample_inds = np.random.randint(0, len(valid_inds[0]), n_samples)

    rand_inds = [v[sample_inds] for v in valid_inds]

    res = np.stack([data[r[0] - patch_size[0] // 2:r[0] + patch_size[0] - patch_size[0] // 2, r[1] - patch_size[1] // 2:r[1] + patch_size[1] - patch_size[1] // 2] for r in zip(*rand_inds)])

    return res

def sample_patches_from_img(coords, img):
    assert coords[:,0].max() <= img.shape[0]-x_width
    assert coords[:,1].max() <= img.shape[1]-y_width
    patches = np.zeros(shape=(coords.shape[0], x_width, y_width), dtype=img.dtype)
    for m,ind in enumerate(coords):
        patches[m] = img[ind[0]:ind[0]+x_width, ind[1]:ind[1]+y_width]
    return patches

def random_patch_coords(img, n):
    xc = np.random.randint(img.shape[0]-x_width, size=n)
    yc = np.random.randint(img.shape[1]-y_width, size=n)
    return np.stack((xc, yc), axis=1)

def regular_patch_coords(img):
    dx,dy = img.shape[0]-x_width, img.shape[1]-y_width
    coords = skut.view_as_windows(np.indices((dx+1,dy+1)), (2,1,1), step=step)
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

def rebuild_img_from_patch_activations(x_y, patchact, coords):
    "TODO: potentially add more ways of recombining than a simple average, i.e. maximum"
    # TODO: this will break.
    x,y = x_y
    n_samp, dx, dy, nclasses = patchact.shape
    zeros_img = np.zeros(shape=(x,y,nclasses))
    count_img = np.zeros(shape=(x,y,nclasses))
    print("Nans?: ", np.sum(map(util.count_nans, patchact)))
    for cord,patch in zip(coords, patchact):
        x,y = cord
        zeros_img[x:x+dx, y:y+dy] += patch
        count_img[x:x+dx, y:y+dy] += np.ones_like(patch)
        # z = zeros_img[x:x+dx, y:y+dy]
        # zeros_img[x:x+dx, y:y+dy] = np.where(z>patch, z, patch)
        # assert 0>1
    print(map(util.count_nans, [zeros_img, count_img]))
    # res = zeros_img/count_img
    # res[res==np.nan] = -1
    return zeros_img/count_img
    # return zeros_img

def imglist_to_X(greylist):
    """turn list of images into ndarray of patches, labels and their coordinates. Used for
    both training and testing."""

    coords = map(regular_patch_coords, greylist)
    f = lambda c_g: sample_patches_from_img(c_g[0],c_g[1])
    greypatches = map(f, zip(coords, greylist))
    X = np.concatenate(tuple(greypatches), axis=0)

    # reshape into theano dimension ordering
    a,b,c = X.shape
    assert K.image_dim_ordering() == 'th'
    X = X.reshape(a, 1, b, c)

    # normalize X per patch
    mi = np.amin(X,axis = (1,2,3), keepdims = True)
    ma = np.amax(X,axis = (1,2,3), keepdims = True)+1.e-10
    X = (X-mi)/(ma-mi)
    return X.astype(np.float32)

def imglist_to_Y(labellist):
    "turn list of images into ndarray of patches, labels and their coordinates"

    coords = map(regular_patch_coords, labellist)
    f = lambda c_g: sample_patches_from_img(c_g[0],c_g[1])
    labelpatches = map(f, zip(coords, labellist))
    Y = np.concatenate(tuple(labelpatches), axis=0)

    # reshape into theano dimension ordering
    a,b,c = Y.shape
    assert K.image_dim_ordering() == 'th'
    Y = Y.reshape(a, 1, b, c)

    # convert label values to vector of label scores
    # Y[Y==2] = 1
    Y[Y!=0] = 3
    Y[Y==0] = 1
    Y[Y==3] = 0
    assert Y.min() == 0
    nb_classes = Y.max()+1
    Y = Y.reshape(a*b*c)
    Y = np_utils.to_categorical(Y, nb_classes)
    Y = Y.reshape(a, b*c, nb_classes)
    return Y.astype(np.float32)

def imglists_to_XY(greylist, labellist):
    X = imglist_to_X(greylist)
    Y = imglist_to_Y(labellist)
    return X,Y

def process_XY_for_training(X,Y):
    assert X.ndim==4 # samples, y, x + channels...
    a,b,c,d = X.shape
    X = X.reshape(a,c,d)
    Y = Y.reshape(a,c,d,2)
    ylabel = np.argmax(Y, axis=-1)
    # 0.15 looks like a good value after close image inspection
    # inds1 = np.mean(X, axis=(1,2)) > 0.15
    inds = np.any(ylabel==1, axis=(1,2))
    X = X[inds]
    Y = Y[inds]
    a,c,d = X.shape
    X1 = np.flipud(X)
    X2 = np.fliplr(X)
    Y1 = np.flipud(Y)
    Y2 = np.fliplr(Y)
    X3 = np.flipud(np.fliplr(X))
    Y3 = np.flipud(np.fliplr(Y))
    X = np.concatenate((X, X1, X2, X3), axis=0)
    Y = np.concatenate((Y, Y1, Y2, Y3), axis=0)
    # from scipy.ndimage import rotate
    # x_rotations = [rotate(X, theta, axes=(1,2), reshape=False) for theta in [-10, 0, 10]]
    # y_rotations = [rotate(Y, theta, axes=(1,2), reshape=False, order=0) for theta in [-10, 0, 10]]
    # X = np.concatenate(tuple(x_rotations), axis=0)
    # Y = np.concatenate(tuple(y_rotations), axis=0)
    X = X.reshape(4*a,1,c,d)
    Y = Y.reshape(4*a,c*d,2)
    return X,Y

# setup and train the model

def my_categorical_crossentropy(weights =(1., 1.)):
    def catcross(y_true, y_pred):
        return -(weights[0] * K.mean(y_true[:,:,0]*K.log(y_pred[:,:,0]+K.epsilon())) +
                 weights[1] * K.mean(y_true[:,:,1]*K.log(y_pred[:,:,1]+K.epsilon())))

        # return -(K.mean(y_true[:,:,0]*K.log(y_pred[:,:,0]+K.epsilon()))+K.mean(y_true[:,:,1]*K.log(y_pred[:,:,1]+K.epsilon())))
    return catcross

def get_unet():
    """
    The information travel distance gives a window of 29 pixels square.
    """
    inputs = Input((1, y_width, x_width))
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
    conv6 = core.Reshape((2, y_width*x_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)
    return model

def get_unet_2():
    inputs = Input((1, y_width, x_width))
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
    conv6 = core.Reshape((2, y_width*x_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)
    return model

def trainmodel(X_train, Y_train, X_vali, Y_vali, model=None, batch_size = 128, nb_epoch = 1, patience = 5, savedir=None):
    """
    Note: Input X,Y should really just be training data! Not all the labeled data we have!
    """

    # Adjust Sample weights
    classimg = np.argmax(Y_train, axis=-1).flatten()
    n_zeros = len(classimg[classimg==0])
    n_ones = len(classimg[classimg==1])
    classweights = {0: 1, 1: n_zeros/n_ones}
    print(classweights)

    # Which model?
    if model is None:
        model = get_unet()

    # How to optimize?
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    # model.compile(optimizer='sgd',loss=my_categorical_crossentropy((1,30.)),metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr = 0.001),loss=my_categorical_crossentropy((1,10.)),metrics=['mean_squared_error'])
    cw = classweights
    model.compile(optimizer=Adam(lr = 0.0005), loss=my_categorical_crossentropy(weights=(cw[0], 10*cw[1])), metrics=['accuracy'])

    # Callbacks
    # TODO: IO/Filepaths controlled by single module...
    if savedir is None:
        checkpointer = ModelCheckpoint(filepath="./unet_model_weights_checkpoint.h5", verbose=0, save_best_only=True, save_weights_only=True)
    else:
        checkpointer = ModelCheckpoint(filepath=savedir + "/unet_model_weights_checkpoint.h5", verbose=0, save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(patience=patience, verbose=0)
    callbacks = [checkpointer, earlystopper]

    # Build and Train
    model.fit(X_train,
              Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_vali, Y_vali),
              callbacks=callbacks)

    score = model.evaluate(X_vali, Y_vali, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model


def trainmodel_generator(X_train, Y_train, X_vali, Y_vali, model, datagen, learning_rate = 0.0005, membrane_weight_multiplier=10, batch_size = 128, nb_epoch = 1, patience = 5, savedir="./"):
    """
    Note: Input X,Y should really just be training data! Not all the labeled data we have!
    """

    # Adjust Sample weights
    classimg = np.argmax(Y_train, axis=-1).flatten()
    n_zeros = len(classimg[classimg==0])
    n_ones = len(classimg[classimg==1])
    classweights = {0: 1, 1: n_zeros/n_ones}
    print(classweights)

    # How to optimize?
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    # model.compile(optimizer='sgd',loss=my_categorical_crossentropy((1,30.)),metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr = 0.001),loss=my_categorical_crossentropy((1,10.)),metrics=['mean_squared_error'])
    cw = classweights
    model.compile(optimizer=Adam(lr = learning_rate), loss=my_categorical_crossentropy(weights=(cw[0], membrane_weight_multiplier*cw[1])), metrics=['accuracy'])

    # Callbacks
    # TODO: IO/Filepaths controlled by single module...
    checkpointer = ModelCheckpoint(filepath=savedir + "/unet_model_weights_checkpoint.h5", verbose=1, save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(patience=patience, verbose=1)
    callbacks = [checkpointer, earlystopper]

    # Build and Train
    model.fit_generator(
              datagen.flow(X_train, Y_train, batch_size=batch_size),
              samples_per_epoch=X_train.shape[0],
              nb_epoch=nb_epoch,
              verbose=1,
              validation_data=datagen.flow(X_vali, Y_vali),
              nb_val_samples=X_vali.shape[0],
              callbacks=callbacks)

    score = model.evaluate(X_vali, Y_vali, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model



# use the model for prediction

def predict_single_image(model, img, batch_size=4):
    "unet predict on a greyscale img"
    X = imglist_to_X([img])
    Y_pred = model.predict(X, batch_size=batch_size)
    a,b,c = Y_pred.shape
    assert c==2

    Y_pred = Y_pred.reshape((a, y_width, x_width, c))
    # WARNING TODO: This will break when we change the coords used in `imglist_to_X`
    coords = regular_patch_coords(img)
    res = rebuild_img_from_patch_activations(img.shape, Y_pred, coords)
    return res[:,:,1].astype(np.float32)
