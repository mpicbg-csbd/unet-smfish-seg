import sys
import unet
from skimage.io import imread
import datasets as d
import util
import json
import numpy as np

rationale = """
Check out predictions of m90 on original sized images! With fixed boundary effect regions!
"""

predict_params = {
 'savedir' : './',
 'model_weights' : 'training/m129/unet_model_weights_checkpoint.h5',
 'grey_tif_folder' : "data3/labeled_data_cellseg/greyscales/",
 'x_width' : 480,
 'y_width' : 480,
 'step' : 120,
 'batch_size' : 4,
 'n_convolutions_first_layer' : 32,
 'dropout_fraction' : 0.2,
 'itd' : None,
 'model' : 'unet_7layer',
}

def predict(predict_params):
    predict_image_names = util.sglob(predict_params['grey_tif_folder'] + '*.tif')

    unet.savedir = predict_params['savedir']
    unet.x_width = predict_params['x_width']
    unet.y_width = predict_params['y_width']
    unet.step = predict_params['step']

    if predict_params['model'] == 'unet_7layer':
        model = unet.get_unet_7layer()
        unet.itd = 44
        predict_params['itd'] = 44
    elif predict_params['model'] == 'unet_5layer':
        model = unet.get_unet()
        unet.itd = 20
        predict_params['itd'] = 20

    model.load_weights(predict_params['model_weights'])

    for name in predict_image_names:
        img = d.imread(name)
        print(name, img.shape)
        res = unet.predict_single_image(model, img, batch_size=predict_params['batch_size'])
        combo = np.stack((img, res), axis=0)
        path, base, ext =  util.path_base_ext(name)
        d.imsave(predict_params['savedir'] + "/" + base + '_predict' + ext, combo.astype('float32'))

if __name__ == '__main__':
    predict_params['savedir'] = sys.argv[1]
    predict(predict_params)

