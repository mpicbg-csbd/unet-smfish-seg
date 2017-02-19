'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

import skimage.util as ut

# input image dimensions
img_xdim, img_ydim = 28, 28
# TODO: make stride x,y dependent
stride = 10
input_shape = (img_xdim, img_ydim, 1)
nb_classes = 3

def splt(img):
    """split and image into a vector of patch squares."""
    x,y = img.shape
    print("x,y: ", x, y)
    if not np.remainder(x-img_xdim,stride)==0:
        print("FAIL, STRIDE DOESN'T FIT")
    if not np.remainder(y-img_ydim,stride)==0:
        print("FAIL, STRIDE DOESN'T FIT")
    img = ut.view_as_windows(img, (img_xdim,img_ydim), stride)
    a,b,x,y = img.shape
    return img.reshape((a*b,x,y))

def imglists_to_XY(greylist, labellist):
    greypatches = map(splt, greylist)
    X = np.concatenate(tuple(greypatches), axis=0)
    labelpatches = map(splt, labellist)
    labelpatches = np.concatenate(tuple(labelpatches), axis=0)
    # WARNING: ONLY PREDICT THE CENTER PIXEL
    Y = labelpatches[:,14,14]
    global nb_classes
    if K.image_dim_ordering() == 'th':
        X = X.reshape(X.shape[0], 1, img_xdim, img_ydim)
    else:
        X = X.reshape(X.shape[0], img_xdim, img_ydim, 1)
    # convert class vectors to binary class matrices
    Y = np_utils.to_categorical(Y, nb_classes)
    return X, Y

def preprocess(X,Y):
    """Apply to full X and Y, before splitting into training and testing."""

    X = X.astype('float32')
    # TODO: Correct this so that it's normalizing correctly...
    X /= 255.

    return X,Y

def buildmodel():
    # number of convolutional filters to use
    nb_filters = 64
    # nb_filters = 8
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape,
                            activation='relu'))
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], activation='relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], activation='relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model

def trainmodel(X,Y, batch_size = 128, nb_epoch = 3, patience = 5):
    """
    Note: Input X,Y should really just be training data! Not all the labeled data we have!
    """
    import sklearn.utils as ut
    import util
    # from sklearn.model_selection import train_test_split
    split = (X.shape[0]*6)//7

    X_train, Y_train, X_vali, Y_vali = util.train_test_split(X,Y,test_fraction=0.2)
    path_model = "./keras_model.h5"
    classweights = ut.compute_class_weight('balanced', [0,1,2], np.argmax(Y_train, axis=1))
    checkpointer = ModelCheckpoint(filepath=path_model, verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(patience=patience, verbose=0)
    callbacks = [checkpointer, earlystopper]
    classweights = {i:classweights[i] for i in range(len(classweights))}
    print(classweights)
    model = buildmodel()
    model.compile(loss='categorical_crossentropy',
                #   optimizer='adadelta',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_vali, Y_vali), class_weight=classweights,
              callbacks=callbacks)
    # model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #           verbose=1, validation_data=(X_vali, Y_vali), sample_weight=np.array(sampleweights), callbacks=callbacks)

    score = model.evaluate(X_vali, Y_vali, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model

if __name__ == 'main':
    mod = trainmodel()
