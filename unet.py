doc="""
UNET architecture for pixelwise classification
"""

import numpy as np
import skimage.io as io
import json

from keras.activations import softmax
from keras.models import Model
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

nb_classes = 2
learning_rate = 0.0005
membrane_weight_multiplier=10
batch_size = 32
nb_epoch = 100
patience = 5
savedir="./"
samples_per_epoch = 'auto' # 6000

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

    print(map(util.count_nans, [zeros_img, count_img]))
    # res = zeros_img/count_img
    # res[res==np.nan] = -1
    return zeros_img/count_img
    # return zeros_img

def imglist_to_X(greylist):
    """turn list of images into ndarray of patches, labels and their coordinates. Used for
    both training and testing."""

    coords = map(regular_patch_coords, greylist)
    greypatches = [sample_patches_from_img(c,g) for c,g in zip(coords, greylist)]
    X = np.concatenate(tuple(greypatches), axis=0)

    # normalize X per patch
    mi = np.amin(X,axis = (1,2), keepdims = True)
    ma = np.amax(X,axis = (1,2), keepdims = True)+1.e-10
    X = (X-mi)/(ma-mi)
    return X.astype(np.float32)

def imglist_to_Y(labellist):
    "turn list of images into ndarray of patches, labels and their coordinates"

    coords = map(regular_patch_coords, labellist)
    labelpatches = [sample_patches_from_img(c,g) for c,g in zip(coords, labellist)]
    Y = np.concatenate(tuple(labelpatches), axis=0)
    return Y

def labels_to_activations(Y):
    # print "Ymin is :", Y.min()
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
        X = X.reshape(a, 1, b, c)
    elif K.image_dim_ordering() == 'tf':
        X = X.reshape(a, b, c, 1)
    return X

def imglists_to_XY(greylist, labellist):
    X = imglist_to_X(greylist)
    Y = imglist_to_Y(labellist)
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

    print "\n\nK dim orderin is! : ", K.image_dim_ordering(), "\n\n"
    if K.image_dim_ordering() == 'th':
      inputs = Input((1, y_width, x_width))
      concatax = 1
    if K.image_dim_ordering() == 'tf':
      inputs = Input((y_width, x_width, 1))
      concatax = 3

    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)

    up1 = UpSampling2D(size=(2,2))(conv3)
    cat1 = Concatenate(axis=concatax)([up1, conv2])
    conv4 = Conv2D(64, (3, 3), padding='same', activation='relu')(cat1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    cat2 = Concatenate(axis=concatax)([up2, conv1])
    conv5 = Conv2D(32, (3, 3), padding='same', activation='relu')(cat2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv5)

    # here nb_classes used to be just the number 2
    conv6 = Conv2D(2, (1, 1), padding='same', activation='relu')(conv5)
    # conv6 = core.Permute((2,3,1))(conv6)
    softm = lambda x: softmax(x, axis=concatax)
    conv7 = core.Activation(softm)(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    return model


# ---- PUBLIC INTERFACE ----

def train_unet(grey_imgs, label_imgs, model):
    "greys and labels are lists of images."

    # We're training on only the right half of each image!
    # Then we can easily identify overfitting by eye.
    grey_imgs_small = []
    label_imgs_small = []
    for grey,lab in zip(grey_imgs, label_imgs):
        a,b = grey.shape
        grey_imgs_small.append(grey[:,0:b//2])
        label_imgs_small.append(lab[:,0:b//2])

    # create ndarray of patches
    print("CREATING PATCHES\n\n")
    X,Y = imglists_to_XY(grey_imgs_small, label_imgs_small)

    # shuffle the patches
    print("SHUFFLING PATCHES\n\n")
    print("X.shape = ", X.shape, " and Y.shape = ", Y.shape)
    train_ind, test_ind = util.subsample_ind(X, Y, test_fraction=0.2, rand_state=0)
    print("train_ind = ", train_ind, " and test_ind =", test_ind)
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    X_train, Y_train, X_vali, Y_vali = X[train_ind], Y[train_ind], X[test_ind], Y[test_ind]

    # Adjust sample weights
    print("SETUP THE CLASSWEIGHTS\n\n")
    classimg = np.argmax(Y_train, axis=-1).flatten()
    n_zeros = len(classimg[classimg==0])
    n_ones = len(classimg[classimg==1])
    classweights = {0: 1, 1: n_zeros/n_ones}
    print(classweights)
    cw = classweights
    model.compile(optimizer=Adam(lr = learning_rate), loss=my_categorical_crossentropy(weights=(cw[0], membrane_weight_multiplier*cw[1])), metrics=['accuracy'])

    # Setup callbacks
    print("SETUP CALLBACKS\n\n")
    checkpointer = ModelCheckpoint(filepath=savedir + "/unet_model_weights_checkpoint.h5", verbose=1, save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(patience=patience, verbose=1)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    # callbacks = [checkpointer, earlystopper]
    callbacks = [checkpointer]

    X_vali = add_singleton_dim(X_vali)
    Y_vali = labels_to_activations(Y_vali)

    # import datasets as d

    # g = batch_generator_patches(X_train, Y_train)
    # for i in range(10):
    #     xbatch, ybatch = X_train[i], Y_train[i]
    #     d.imsave("xbatch{}.tif".format(i), xbatch)
    #     d.imsave("ybatch{}.tif".format(i), ybatch)

    # return "poopy"

    # # Build and Train
    # print("RUN FIT GENERATOR")

    # g = batch_generator_patches(X_train, Y_train)
    # for i in range(10):
    #     xbatch, ybatch = next(g)
    #     d.imsave("xbatch{}.tif".format(i), xbatch)
    #     d.imsave("ybatch{}.tif".format(i), ybatch)

    # return "nothing"

    global samples_per_epoch
    if samples_per_epoch == 'auto':
        samples_per_epoch, _ = divmod(X_train.shape[0], batch_size)

    history = model.fit_generator(
                batch_generator_patches(X_train, Y_train),
                samples_per_epoch=samples_per_epoch,
                #samples_per_epoch=X_train.shape[0],
                nb_epoch=nb_epoch,
                verbose=1,
                validation_data=(X_vali, Y_vali),
                #nb_val_samples=X_vali.shape[0],
                callbacks=callbacks)

    json.dump(history.history, open(savedir + '/history.json', 'w'))
    score = model.evaluate(X_vali, Y_vali, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return history

def batch_generator_patches(X,Y, verbose=False):
    # inds = np.arange(X.shape[0])
    # np.random.shuffle(inds)
    # X = X[inds]
    # Y = Y[inds]
    epoch = 0
    while (True):
        print("INSIDE genertor! Loop Count is: ", epoch, '\n\n')
        epoch += 1
        current_idx = 0
        batchnum = 0
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds]
        Y = Y[inds]
        # while current_idx+batch_size <= X.shape[0]:
        # while current_idx+batch_size <= samples_per_epoch*batch_size:
        while batchnum < samples_per_epoch:
            if verbose:
                print("yielding")
            Xbatch, Ybatch = X[current_idx:current_idx+batch_size].copy(), Y[current_idx:current_idx+batch_size].copy()
            current_idx += batch_size

            # if we're gonna augment, do it here... (applied to both training and validation patches!)
            # e.g. flip and rotate images randomly

            Xbatch = add_singleton_dim(Xbatch)
            Ybatch = labels_to_activations(Ybatch)

            #print("Yielding X,Y. Size and Shape: ")
            #print(Xbatch.shape, Ybatch.shape)
            batchnum += 1
            yield Xbatch, Ybatch


# use the model for prediction

def predict_single_image(model, img, batch_size=4):
    "unet predict on a greyscale img"
    X = imglist_to_X([img])
    X = add_singleton_dim(X)
    Y_pred = model.predict(X, batch_size=batch_size)
    # io.imsave('ypred.tif', Y_pred)
    print("Y_pred shape: ", Y_pred.shape)
    a,b,c,d = Y_pred.shape
    assert d==2
    # Y_pred = Y_pred.reshape((a, y_width, x_width, c))
    # WARNING TODO: This will break when we change the coords used in `imglist_to_X`
    coords = regular_patch_coords(img)
    res = rebuild_img_from_patch_activations(img.shape, Y_pred, coords)
    return res[:,:,1].astype(np.float32)
