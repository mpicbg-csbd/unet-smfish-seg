import sys
import ipy

if __name__ == '__main__':
    greys = ipy.greyscales_down3x()
    labels = ipy.labels_down3x()
    ipy.unet.x_width = 160
    ipy.unet.y_width = 160
    ipy.unet.step = 10
    model = ipy.unet.get_unet()
    # model.load_weights("unet_params/unet_model_weights_checkpoint_120patch10stride.h5")
    # model.load_weights("results/trial0011/unet_model_weights_checkpoint.h5")
    # saveDir = sys.argv[1]
    ipy.train_unet(greys, labels, model, saveDir)
    # ipy.predict_unet(ipy.unseen_greys(), model)