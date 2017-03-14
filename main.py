import sys
import ipy
from util import sglob

if __name__ == '__main__':
    greys = sglob("data2/greyscales/down6x/*.tif")
    labels = sglob("data2/Cell_segmentations_paper/pooled6/*.tif")
    # info_travel_dist = 14
    # window_width = 2*info_travel_dist + 1 approx 30
    ipy.unet.x_width = 120
    ipy.unet.y_width = 120
    ipy.unet.step = 30
    model = ipy.unet.get_unet()
    # model.load_weights("unet_params/unet_model_weights_checkpoint_120patch10stride.h5")
    # model.load_weights("results/trial0011/unet_model_weights_checkpoint.h5")
    # model.load_weights("results/halfhalf/unet_model_weights_checkpoint.h5")
    # model.load_weights("results/halfhalf/unet_model_weights_checkpoint.h5")
    saveDir = sys.argv[1]
    ipy.train_unet(greys, labels, model, saveDir)
    # ipy.predict_unet(greys, model, saveDir)
