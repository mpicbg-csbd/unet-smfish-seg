import sys
import ipy

if __name__ == '__main__':
    print(ipy.knime_train_data_greys())
    print(ipy.knime_train_data_labels())
    ipy.unet.x_width = 160
    ipy.unet.y_width = 160
    ipy.unet.step = 10
    model = ipy.unet.get_unet()
    model.load_weights("unet_params/unet_model_weights_checkpoint_120patch10stride.h5")
    saveDir = sys.argv[1]
    ipy.train_unet(ipy.knime_train_data_greys(), ipy.knime_train_data_labels(), model, saveDir)
