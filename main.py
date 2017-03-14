import sys
sys.path.append("./models/")
import unet
from util import sglob
from skimage.io import imread
import datasets as d

if __name__ == '__main__':
    greys = sglob("data2/greyscales/down3x/*.tif")
    greys = [d.imread(img) for img in greys]
    labels = sglob("data2/labels/down3x/*.tif")
    labels = [d.imread(img) for img in labels]

    print("Input greyscale images:")
    for name in greys:
        print(name)
    print("Input label images:")
    for name in labels:
        print(name)

    # info_travel_dist = 14
    # window_width = 2*info_travel_dist + 1 approx 30
    saveDir = sys.argv[1]
    unet.x_width = 120
    unet.y_width = 120
    unet.step = 30
    unet.batch_size = 10
    unet.savedir = saveDir
    model = unet.get_unet()
    # model.load_weights("unet_params/unet_model_weights_checkpoint_120patch10stride.h5")
    # model.load_weights("results/trial0011/unet_model_weights_checkpoint.h5")
    # model.load_weights("results/halfhalf/unet_model_weights_checkpoint.h5")
    # model.load_weights("results/halfhalf/unet_model_weights_checkpoint.h5")
    unet.train_unet(greys, labels, model)
    # ipy.predict_unet(greys, model, saveDir)
