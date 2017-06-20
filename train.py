import sys
import unet
from skimage.io import imread
import datasets as d
import util
import time
import json

rationale = """
Continuation of m43, which continues from m39... Just the same thing but for 500 more epochs.
"""

train_params = {
 'savedir' : './',
 'grey_tif_folder' : "data3/labeled_data_cellseg/greyscales/down6x/",
 'label_tif_folder' : "data3/labeled_data_cellseg/labels/down6x/",
 'initial_model_params' : "training/m43/unet_model_weights_checkpoint.h5",
 'x_width' : 120,
 'y_width' : 120,
 'step' : 30,
 'batch_size' : 32,
 'learning_rate' : 5e-5,
 'membrane_weight_multiplier' : 1,
 'epochs' : 500
}


def train(train_params):
    start_time = time.time()

    train_grey_names = []
    train_grey_imgs = []
    train_label_imgs = []

    grey_names = util.sglob(train_params['grey_tif_folder'] + "*.tif")
    label_names = util.sglob(train_params['label_tif_folder'] + "*.tif")
    grey_imgs = [d.imread(img) for img in grey_names]
    label_imgs = [d.imread(img) for img in label_names]

    ## remove cell-type channel
    if 'data3/labeled_data_cellseg/greyscales/' == train_params['label_tif_folder']:
        label_imgs = [img[0] for img in label_imgs]

    ## make 0-valued membrane 1-valued (for cellseg labels only)
    if 'labeled_data_cellseg' in train_params['label_tif_folder']:
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
    train_grey_imgs += grey_imgs
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
    # unet.steps_per_epoch = train_params['steps_per_epoch']

    model = unet.get_unet()
    if train_params['initial_model_params']:
        model.load_weights(train_params['initial_model_params'])

    begin_training_time = time.time()
    history = unet.train_unet(train_grey_imgs, train_label_imgs, model)
    finished_time = time.time()

    history.history['warm_up_time'] = begin_training_time - start_time
    history.history['train_time'] = finished_time - begin_training_time
    json.dump(history.history, open(train_params['savedir'] + '/history.json', 'w'))
    train_params['rationale'] = rationale
    json.dump(train_params, open(train_params['savedir'] + '/train_params.json', 'w'))
    return history


if __name__ == '__main__':
    train_params['savedir'] = sys.argv[1]
    train(train_params)
