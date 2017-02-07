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

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
nb_classes = 10

def prepare_input():
    # the data, shuffled and split between train and test sets
    (X_train, Y_train0), (X_test, Y_test0) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train0, nb_classes)
    Y_test = np_utils.to_categorical(Y_test0, nb_classes)
    return (X_train, Y_train), (X_test, Y_test)

def prepare_input_fish():
    # the data, shuffled and split between train and test sets
    import keras_classifier as kc
    (X_train, Y_train0), (X_test, Y_test0) = kc.buildTrainingData()

    global input_shape, nb_classes
    input_shape = (28, 28, 1)
    nb_classes = 3

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # X_train /= np.mean(X_train.reshape(len(X_train), -1), axis=-1)
    # X_test /= np.mean(X_train.reshape(len(X_test), -1), axis=-1)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train0, nb_classes)
    Y_test = np_utils.to_categorical(Y_test0, nb_classes)
    # Y_train = Y_train0
    # Y_test = Y_test0
    return (X_train, Y_train), (X_test, Y_test)

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

def trainmodel(batch_size = 128, nb_epoch = 100, patience = 5):
    import sklearn.utils as ut
    (X_train, Y_train), (X_test, Y_test) = prepare_input_fish()
    path_model = "./keras_model.h5"
    classweights = ut.compute_class_weight('balanced', [0,1,2], np.argmax(Y_train, axis=1))
    # sampleweights = ut.compute_sample_weight(classweights, )
    checkpointer = ModelCheckpoint(filepath=path_model, verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(patience=patience, verbose=0)
    callbacks = [checkpointer, earlystopper]
    classweights = {i:classweights[i] for i in range(len(classweights))}
    # weights = 1./Y_train.sum(axis=0)
    # sampleweights = []
    # for y in np.argmax(Y_train, axis=1):
    #     if y == 0:
    #         sampleweights.append(weights[0])
    #     elif y == 1:
    #         sampleweights.append(weights[1])
    #     elif y == 2:
    #         sampleweights.append(weights[2])
    #     else:
    #         raise Exception()
    print(classweights)
    model = buildmodel()
    model.compile(loss='categorical_crossentropy',
                #   optimizer='adadelta',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test), class_weight=classweights, callbacks=callbacks)
    # model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #           verbose=1, validation_data=(X_test, Y_test), sample_weight=np.array(sampleweights), callbacks=callbacks)
    model = load_model(path_model)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model

if __name__ == 'main':
    mod = trainmodel()
