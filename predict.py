import sys
sys.path.insert(0, "../.local/lib/python3.5/site-packages/")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import util
import json
import numpy as np
from skimage.io import imread
import unet
import datasets as d

rationale = """
Test out predict.py refactor.
"""

predict_params = {
 'savedir' : './',
 'model_weights' : 'training/m162/unet_model_weights_checkpoint.h5',
 'grey_tif_folder' : "data3/labeled_data_cellseg/greyscales/",
 'batch_size' : 4,
}

def predict(predict_params):
    model_dir = os.path.dirname(predict_params['model_weights'])
    train_params = json.load(open(model_dir + '/train_params.json'))
    predict_params['grey_tif_folder'] = train_params['grey_tif_folder']

    model = unet.get_unet_n_pool(train_params['n_pool'], 
                                 train_params['n_convolutions_first_layer'],
                                 train_params['dropout_fraction'])
    print(model.summary())
    model.load_weights(predict_params['model_weights'])

    unet.savedir = train_params['savedir']
    # unet.x_width = train_params['x_width']
    # unet.y_width = train_params['y_width']
    #unet.itd = train_params['itd']
    #unet.step = train_params['step']
    npool = 4
    unet.step = 16*35 # = 2**4 * 10 # *unet.x_width - 2*unet.itd #92 #100 #500 #310
    unet.itd  = 92 #190 #train_params['itd']
    unet.x_width = 16*(35+15)
    unet.y_width = unet.x_width

    predict_image_names = util.sglob(predict_params['grey_tif_folder'] + '*.tif')

    for name in predict_image_names[:5]:
        img = d.imread(name)
        print(name, img.shape)
        res = unet.predict_single_image(model, img, batch_size=predict_params['batch_size'])
        print("Res shape", res.shape)
        combo = np.stack((img, res), axis=0)
        path, base, ext =  util.path_base_ext(name)
        d.imsave(predict_params['savedir'] + "/" + base + '_predict_' + str(unet.itd) + ext, combo.astype('float32'))

if __name__ == '__main__':
    predict_params['savedir'] = sys.argv[1]
    predict(predict_params)

