import sys
# sys.path.append("./models/")
import unet
from skimage.io import imread
import datasets as d
import util
import json


rationale = """
Do predictions for m39 with val_loss ~= 0.0127
"""

predict_params = {
 'savedir' : './',
 'model_weights' : 'training/m39/unet_model_weights_checkpoint.h5',
 'grey_tif_folder' : "data3/labeled_data_cellseg/greyscales/down6x/",
 'x_width' : 120,
 'y_width' : 120,
 'step' : 10,
 'batch_size' : 32,
}

def predict(predict_params):
    predict_image_names = util.sglob(predict_params['grey_tif_folder'] + '*.tif')

    unet.savedir = predict_params['savedir']
    unet.x_width = predict_params['x_width']
    unet.y_width = predict_params['y_width']
    unet.step = predict_params['step']
    model = unet.get_unet()
    model.load_weights(predict_params['model_weights'])

    for name in predict_image_names:
        img = d.imread(name)
        print(name, img.shape)
        res = unet.predict_single_image(model, img, batch_size=predict_params['batch_size'])
        # print("There are {} nans!".format(np.count_nonzero(~np.isnan(res))))
        path, base, ext =  util.path_base_ext(name)
        d.imsave(predict_params['savedir'] + "/" + base + '_predict' + ext, res.astype('float32'))

if __name__ == '__main__':
    predict_params['savedir'] = sys.argv[1]
    predict(predict_params)

