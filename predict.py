import sys
sys.path.insert(0, "../.local/lib/python3.5/site-packages/")
import unet
from skimage.io import imread
import datasets as d
import util
import json
import numpy as np
import os

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
    unet.x_width = train_params['x_width']
    unet.y_width = train_params['y_width']
    #unet.itd = train_params['itd']
    unet.itd = 190 #train_params['itd']
    #unet.step    = train_params['step']
    unet.step    = 100 #500 #310
    
    predict_image_names = util.sglob(predict_params['grey_tif_folder'] + '*.tif')

    for name in predict_image_names[:2]:
        img = d.imread(name)
        print(name, img.shape)
        res = unet.predict_single_image(model, img, batch_size=predict_params['batch_size'])
        combo = np.stack((img, res), axis=0)
        path, base, ext =  util.path_base_ext(name)
        d.imsave(predict_params['savedir'] + "/" + base + '_predict' + ext, combo.astype('float32'))

if __name__ == '__main__':
    predict_params['savedir'] = sys.argv[1]
    predict(predict_params)

