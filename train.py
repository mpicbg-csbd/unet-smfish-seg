import sys
sys.path.insert(0, "/home/broaddus/.local/lib/python3.5/site-packages/")
import pkg_resources
pkg_resources.require("scikit-image>=0.13.0")
import skimage
from skimage.io import imread

import unet
import datasets as d
import util
import time
import json
import numpy as np

rationale = """
SGD. High learn rate. High momentum.
Same as 119==131, but with all the augmentations.
"""

train_params = {
 'savedir' : './',
 'grey_tif_folder' : "data3/labeled_data_cellseg/greyscales/",
 'label_tif_folder' : "data3/labeled_data_cellseg/labels/",
 'x_width' : 480,
 'y_width' : 480,
 'step' : 240,

 'batch_size' : 1,
 'membrane_weight_multiplier' : 1,
 'epochs' : 300,
 'patience' : 30,

 'optimizer' : 'sgd', # 'sgd' or 'adam' (adam ignores momentum)
 'learning_rate' : 1e-3, #3.16e-5,
 'momentum' : 0.99,

 'warping_size' : 10,
 'flipLR' : True,
 'rotate_angle_max' : 30,

 'model' : 'unet_7layer',
 'initial_model_params' : None, #"training/m108/unet_model_weights_checkpoint.h5",
 'n_convolutions_first_layer' : 32,
 'dropout_fraction' : 0.2,
 'itd' : None,
}


def train(train_params):
    start_time = time.time()

    train_grey_names = []
    train_grey_imgs = []
    train_label_imgs = []

    grey_names  = util.sglob(train_params['grey_tif_folder'] + "*.tif")
    label_names = util.sglob(train_params['label_tif_folder'] + "*.tif")
    grey_imgs  = [d.imread(img) for img in grey_names]
    label_imgs = [d.imread(img) for img in label_names]

    ## make 0-valued membrane 1-valued (for cellseg labels only)
    if 'labeled_data_cellseg' in train_params['label_tif_folder']:
        if 'down' not in train_params['label_tif_folder'][-10:]:
            label_imgs = [img[0] for img in label_imgs]
        for img in label_imgs:
            img[img!=0]=2
            img[img==0]=1
            img[img==2]=0
    ## make 2-valued vertex label into 1-valued membrane label
    elif 'labeled_data_membranes' in train_params['label_tif_folder']:
        for img in label_imgs:
            img[img==2] = 1

    ## add to list
    train_grey_names += grey_names
    train_grey_imgs  += grey_imgs
    train_label_imgs += label_imgs

    print("Input greyscale and label images:")
    for n,g,l in zip(train_grey_names, train_grey_imgs, train_label_imgs):
        print(n,g.shape, l.shape)

    # valid training and prediction params (change these before prediction!)
    unet.savedir = train_params['savedir']
    unet.x_width = train_params['x_width']
    unet.y_width = train_params['y_width']
    unet.step = train_params['step']

    # just training params
    unet.batch_size = train_params['batch_size']
    unet.learning_rate = train_params['learning_rate']
    unet.epochs = train_params['epochs']
    unet.membrane_weight_multiplier = train_params['membrane_weight_multiplier']
    unet.patience = train_params['patience']
    unet.n_convolutions_first_layer = train_params['n_convolutions_first_layer']
    unet.dropout_fraction = train_params['dropout_fraction']
    unet.momentum = train_params['momentum']
    unet.warping_size = train_params['warping_size']
    unet.flipLR = train_params['flipLR']
    unet.rotate_angle_max = train_params['rotate_angle_max']

    if train_params['model'] == 'unet_7layer':
        model = unet.get_unet_7layer()
        unet.itd = 44
        train_params['itd'] = 44
    elif train_params['model'] == 'unet_5layer':
        model = unet.get_unet()
        unet.itd = 20
        train_params['itd'] = 20

    print(model.summary())
    if train_params['initial_model_params']:
        model.load_weights(train_params['initial_model_params'])

    begin_training_time = time.time()
    history = unet.train_unet(train_grey_imgs, train_label_imgs, model)
    finished_time = time.time()

    history.history['warm_up_time'] = begin_training_time - start_time
    train_time = finished_time - begin_training_time
    history.history['train_time'] = train_time
    trained_epochs = len(history.history['acc'])
    history.history['trained_epochs'] = trained_epochs
    history.history['avg_time_per_epoch'] = train_time / trained_epochs
    history.history['avg_time_per_batch'] = train_time / (trained_epochs * history.history['steps_per_epoch'])
    history.history['avg_time_per_sample'] = train_time / (trained_epochs * history.history['X_train_shape'][0])
    json.dump(history.history, open(train_params['savedir'] + '/history.json', 'w'))

    train_params['rationale'] = rationale
    json.dump(train_params, open(train_params['savedir'] + '/train_params.json', 'w'))

    for name in grey_names:
        img = d.imread(name)
        print(name, img.shape)
        res = unet.predict_single_image(model, img, batch_size=train_params['batch_size'])
        combo = np.stack((img, res), axis=0)
        path, base, ext = util.path_base_ext(name)
        d.imsave(train_params['savedir'] + "/" + base + '_predict' + ext, combo.astype('float32'))

    return history

if __name__ == '__main__':
    train_params['savedir'] = sys.argv[1]
    train(train_params)
