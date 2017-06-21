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
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras import backend as K
from keras.utils import np_utils
# from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

import skimage.util as skut
import util

# global variables for patch dimensions and stride
x_width = 120
y_width = 120
step = 30

# example param values. Set them in train.py
nb_classes = 2
learning_rate = 0.0005
membrane_weight_multiplier=1
batch_size = 32
epochs = 100
patience = 20
savedir="./"

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
    print("This image contains Nans: ", np.sum(map(util.count_nans, patchact)))

    # ignore parts of the image with boundary effects
    mask = np.ones(patchact[0].shape)
    mm = 10
    mask[:,0:mm] = 0
    mask[:,-mm:] = 0
    mask[0:mm,:] = 0
    mask[-mm:,:] = 0

    for cord,patch in zip(coords, patchact):
        x,y = cord
        zeros_img[x:x+dx, y:y+dy] += patch*mask
        count_img[x:x+dx, y:y+dy] += np.ones_like(patch)*mask
        # z = zeros_img[x:x+dx, y:y+dy]
        # zeros_img[x:x+dx, y:y+dy] = np.where(z>patch, z, patch)
        # assert 0>1

    print(list(map(util.count_nans, [zeros_img, count_img])))
    # res = zeros_img/count_img
    # res[res==np.nan] = -1
    return zeros_img/count_img
    # return zeros_img

def imglist_to_X(greylist):
    """turn list of images into ndarray of patches, labels and their coordinates. Used for
    both training and testing."""

    coords = list(map(regular_patch_coords, greylist))
    greypatches = [sample_patches_from_img(c,g) for c,g in zip(coords, greylist)]
    X = np.concatenate(tuple(greypatches), axis=0)

    # normalize X per patch
    mi = np.amin(X,axis = (1,2), keepdims = True)
    ma = np.amax(X,axis = (1,2), keepdims = True)+1.e-10
    X = (X-mi)/(ma-mi)
    return X.astype(np.float32)

def imglist_to_Y(labellist):
    "turn list of images into ndarray of patches, labels and their coordinates"

    coords = list(map(regular_patch_coords, labellist))
    labelpatches = [sample_patches_from_img(c,g) for c,g in zip(coords, labellist)]
    Y = np.concatenate(tuple(labelpatches), axis=0)
    return Y

def labels_to_activations(Y):
    # print("Ymin is :", Y.min())
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

def imglists_to_XY(greylist, labellist):
    X = imglist_to_X(greylist)
    Y = imglist_to_Y(labellist)
    return X,Y

# setup and train the model

def my_categorical_crossentropy_ndim4(weights =(1., 1.)):
    def catcross(y_true, y_pred):
        return -(weights[0] * K.mean(y_true[:,:,:,0]*K.log(y_pred[:,:,:,0]+K.epsilon())) +
                 weights[1] * K.mean(y_true[:,:,:,1]*K.log(y_pred[:,:,:,1]+K.epsilon())))

        # return -(K.mean(y_true[:,:,0]*K.log(y_pred[:,:,0]+K.epsilon()))+K.mean(y_true[:,:,1]*K.log(y_pred[:,:,1]+K.epsilon())))
    return catcross

def my_categorical_crossentropy(weights =(1., 1.)):
    def catcross(y_true, y_pred):
        return -(weights[0] * K.mean(y_true[:,:,0]*K.log(y_pred[:,:,0]+K.epsilon())) +
                 weights[1] * K.mean(y_true[:,:,1]*K.log(y_pred[:,:,1]+K.epsilon())))

        # return -(K.mean(y_true[:,:,0]*K.log(y_pred[:,:,0]+K.epsilon()))+K.mean(y_true[:,:,1]*K.log(y_pred[:,:,1]+K.epsilon())))
    return catcross

def my_categorical_crossentropy_np(weights =(1., 1.)):
    def catcross(y_true, y_pred):
        return -(weights[0] * np.mean(y_true[:,:,0]*np.log(y_pred[:,:,0]+K.epsilon())) +
                 weights[1] * np.mean(y_true[:,:,1]*np.log(y_pred[:,:,1]+K.epsilon())))

        # return -(K.mean(y_true[:,:,0]*K.log(y_pred[:,:,0]+K.epsilon()))+K.mean(y_true[:,:,1]*K.log(y_pred[:,:,1]+K.epsilon())))
    return catcross

def get_unet():
    """
    The information travel distance is 14.
    """

    print("\n\nK dim orderin is! : ", K.image_dim_ordering(), "\n\n")

    if K.image_dim_ordering() == 'th':
      inputs = Input((1, y_width, x_width))
      concatax = 1
      chan = 'channels_first'
    if K.image_dim_ordering() == 'tf':
      inputs = Input((y_width, x_width, 1))
      concatax = 3
      chan = 'channels_last'

    def Conv(w):
        return Conv2D(w, (3,3), padding='same', data_format=chan, activation='relu')
    Pool = MaxPooling2D(pool_size=(2,2), data_format=chan)
    Upsa = UpSampling2D(size=(2,2), data_format=chan)

    ## Begin U-net
    conv1 = Conv(32)(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv(32)(conv1)
  
    pool1 = Pool(conv1)

    conv2 = Conv(64)(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv(64)(conv2)

    pool2 = Pool(conv2)

    conv3 = Conv(128)(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv(128)(conv3)

    up1   = Upsa(conv3)
    cat1  = Concatenate(axis=concatax)([up1, conv2])

    conv4 = Conv(64)(cat1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv(64)(conv4)

    up2   = Upsa(conv4)
    cat2  = Concatenate(axis=concatax)([up2, conv1])

    conv5 = Conv(32)(cat2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv(32)(conv5)

    conv6 = Conv2D(2, (1, 1), padding='same', data_format=chan, activation='relu')(conv5)
    softm = lambda x: softmax(x, axis=concatax)
    conv7 = core.Activation(softm)(conv6)

    if K.image_dim_ordering() == 'th':
        conv7 = core.Permute((2,3,1))(conv7)

    model = Model(inputs=inputs, outputs=conv7)
    return model

def get_unet_mix():
    """
    The information travel distance gives a window of 29 pixels square.
    """
    if K.image_dim_ordering() == 'th':
        inputs = Input((1, y_width, x_width))
        concatax = 1
        chan = 'channels_first'
    if K.image_dim_ordering() == 'tf':
        inputs = Input((y_width, x_width, 1))
        concatax = 3
        chan = 'channels_last'

    conv1 = Conv2D(32, (3, 3), padding='same', data_format=chan, activation='relu')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same', data_format=chan, activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format=chan)(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', data_format=chan, activation='relu')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same', data_format=chan, activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format=chan)(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', data_format=chan, activation='relu')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same', data_format=chan, activation='relu')(conv3)

    up1 = UpSampling2D(size=(2,2), data_format=chan)(conv3)
    cat1 = Concatenate(axis=concatax)([up1, conv2])
    conv4 = Conv2D(64, (3, 3), padding='same', data_format=chan, activation='relu')(cat1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), padding='same', data_format=chan, activation='relu')(conv4)

    up2 = UpSampling2D(size=(2, 2), data_format=chan)(conv4)
    cat2 = Concatenate(axis=concatax)([up2, conv1])

    conv5 = Conv2D(32, (3, 3), padding='same', data_format=chan, activation='relu')(cat2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), padding='same', data_format=chan, activation='relu')(conv5)

    conv6 = Conv2D(2, (1, 1), padding='same', data_format=chan, activation='relu')(conv5)
    if K.image_dim_ordering() == 'th':
        conv6 = core.Reshape((2, y_width*x_width))(conv6)
        conv6 = core.Permute((2,1))(conv6)
    elif K.image_dim_ordering() == 'tf':
        conv6 = core.Reshape((y_width*x_width, 2))(conv6)
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)
    return model

def get_unet_old():
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

# ---- PUBLIC INTERFACE ----

def train_unet(grey_imgs, label_imgs, model):
    "greys and labels are lists of images."

    # We're training on only the right half of each image!
    # Then we can easily identify overfitting by eye.

    print("CREATING NDARRAY PATCHES")
    grey_imgs_small = []
    label_imgs_small = []
    for grey,lab in zip(grey_imgs, label_imgs):
        a,b = grey.shape
        grey_imgs_small.append(grey[:,0:b//2])
        label_imgs_small.append(lab[:,0:b//2])
    X,Y = imglists_to_XY(grey_imgs_small, label_imgs_small)
    # X,Y = imglists_to_XY(grey_imgs, label_imgs)

    print("SPLIT INTO TRAIN AND TEST")
    print("X.shape = ", X.shape, " and Y.shape = ", Y.shape)
    train_ind, test_ind = util.subsample_ind(X, Y, test_fraction=0.2, rand_state=0)
    print("train_ind = ", train_ind, " and test_ind =", test_ind)
    np.save(savedir + '/train_ind.npy', train_ind)
    np.save(savedir + '/test_ind.npy', test_ind)
    X_train, Y_train, X_vali, Y_vali = X[train_ind], Y[train_ind], X[test_ind], Y[test_ind]

    print("SETUP THE CLASSWEIGHTS")
    # IMPORTANT! The weight for membrane is given by the fraction of non-membrane! (and vice versa)
    classimg = Y_train.flatten()
    non_zeros = len(classimg[classimg!=0])
    non_ones = len(classimg[classimg!=1]) * membrane_weight_multiplier
    total = non_zeros + non_ones
    w0 = non_zeros / total
    w1 = non_ones / total
    class_relative_frequncies = {0: w0, 1: w1}
    print(class_relative_frequncies)

    model.compile(optimizer=Adam(lr = learning_rate), loss=my_categorical_crossentropy_ndim4(weights=(w0, w1)), metrics=['accuracy'])

    # Setup callbacks
    print("SETUP CALLBACKS")
    checkpointer = ModelCheckpoint(filepath=savedir + "/unet_model_weights_checkpoint.h5", verbose=0, save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(patience=patience, verbose=0)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks = [checkpointer, earlystopper]

    steps_per_epoch, _ = divmod(X_train.shape[0], batch_size)

    history = model.fit_generator(
                batch_generator_patches(X_train, Y_train, steps_per_epoch),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
                validation_data=(add_singleton_dim(X_vali), labels_to_activations(Y_vali)))

    print("FINISHED TRAINING")

    history.history['steps_per_epoch'] = steps_per_epoch
    history.history['X_train_shape'] = X_train.shape
    history.history['X_vali_shape'] = X_vali.shape

    Y_pred_train = model.predict(add_singleton_dim(X_train), batch_size)
    Y_pred_vali = model.predict(add_singleton_dim(X_vali), batch_size)

    print("ALL THE SHAPES")
    print(X_train.shape, Y_train.shape, Y_pred_train.shape)
    print(X_vali.shape, Y_vali.shape, Y_pred_vali.shape)

    def savetiff(fname, img):
        io.imsave(fname, img, plugin='tifffile', compress=1)

    if Y_pred_train.ndim == 3:
        print("NDIM 3, ")
        Y_pred_train = Y_pred_train.reshape((-1, y_width, x_width, 2))
        Y_pred_vali = Y_pred_vali.reshape((-1, y_width, x_width, 2))
    # savetiff(savedir + '/X_train.tif', X_train)
    # savetiff(savedir + '/Y_train.tif', Y_train)
    # savetiff(savedir + '/Y_pred_train.tif', Y_pred_train)
    # savetiff(savedir + '/X_vali.tif', X_vali)
    # savetiff(savedir + '/Y_vali.tif', Y_vali)
    # savetiff(savedir + '/Y_pred_vali.tif', Y_pred_vali)
    res = np.stack((X_train, Y_train, Y_pred_train[...,1]), axis=-1)
    savetiff(savedir + '/training.tif', res)
    res = np.stack((X_vali, Y_vali, Y_pred_vali[...,1]), axis=-1)
    savetiff(savedir + '/testing.tif', res)

    return history

def batch_generator_patches(X,Y, steps_per_epoch, verbose=False):
    epoch = 0
    while (True):
        epoch += 1
        current_idx = 0
        batchnum = 0
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds]
        Y = Y[inds]
        while batchnum < steps_per_epoch:
            Xbatch, Ybatch = X[current_idx:current_idx+batch_size].copy(), Y[current_idx:current_idx+batch_size].copy()
            current_idx += batch_size

            # if we're gonna augment, do it here... (applied to both training and validation patches!)
            # e.g. flip and rotate images randomly

            Xbatch = add_singleton_dim(Xbatch)
            Ybatch = labels_to_activations(Ybatch)

            batchnum += 1
            yield Xbatch, Ybatch


# use the model for prediction

def predict_single_image(model, img, batch_size=32):
    "unet predict on a greyscale img"
    X = imglist_to_X([img])
    X = add_singleton_dim(X)
    Y_pred = model.predict(X, batch_size=batch_size)
    print("Y_pred shape: ", Y_pred.shape)

    if Y_pred.ndim == 3:
        print("NDIM 3, ")
        Y_pred = Y_pred.reshape((-1, y_width, x_width, 2))

    # WARNING TODO: This will break when we change the coords used in `imglist_to_X`
    coords = regular_patch_coords(img)
    res = rebuild_img_from_patch_activations(img.shape, Y_pred, coords)
    return res[:,:,1].astype(np.float32)
