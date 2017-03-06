import ipy
import sys
import subprocess
import platform

if __name__ == '__main__':
    if platform.uname()[1].startswith('myers-mac-10'):
      print("Normal Sub. Nothing to load.")
    elif platform.uname()[1].startswith('falcon1'):
      subprocess.call("module load hdf5", shell=True)
      subprocess.call("module load gcc/5.3.0", shell=True)
      subprocess.call("module load cuda/8.0.44", shell=True)
    elif platform.uname()[1].startswith('falcon'):
      print("On Madmax. Trying bsub. TODO...")
    else:
      print("ERROR: Couldn't detect platform!")

    print(ipy.knime_train_data_greys())
    print(ipy.knime_train_data_labels())
    ipy.unet.x_width = 160
    ipy.unet.y_width = 160
    ipy.unet.step = 10
    model = ipy.unet.get_unet()
    model.load_weights("unet_params/unet_model_weights_checkpoint_120patch10stride.h5")
    saveDir = sys.argv[1]
    ipy.train_unet(ipy.knime_train_data_greys(), ipy.knime_train_data_labels(), model, saveDir)
    #print(io.find_available_plugins())