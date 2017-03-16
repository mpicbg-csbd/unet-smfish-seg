import sys
# sys.path.append("./models/")
import unet
from skimage.io import imread
import datasets as d
import util

def main(saveDir):
    greys = util.sglob("data3/labeled_data_membranes/images_big/smaller6x/*.tif")
    greys_imgs = [d.imread(img) for img in greys]
    labels = util.sglob("data3/labeled_data_membranes/labels_big/smaller6x/*.tif")
    labels_imgs = [d.imread(img) for img in labels]

    for img in labels_imgs:
        img[img==2]=1

    print("Input greyscale images:")
    for name in greys:
        print(name)
    print("Input label images:")
    for name in labels:
        print(name)

    # info_travel_dist = 14
    # window_width = 2*info_travel_dist + 1 approx 30
    unet.savedir = saveDir
    unet.x_width = 120
    unet.y_width = 120
    unet.step = 30
    unet.batch_size = 32
    unet.learning_rate = 0.0005
    unet.nb_epoch = 300
    model = unet.get_unet()
    # model.load_weights("unet_params/unet_model_weights_checkpoint_120patch10stride.h5")
    # model.load_weights("results/trial0011/unet_model_weights_checkpoint.h5")
    # model.load_weights("results/halfhalf/unet_model_weights_checkpoint.h5")
    # model.load_weights("results/lr_continued/unet_model_weights_checkpoint.h5")
    # model.load_weights("results/lr_0005/unet_model_weights_checkpoint.h5")
    model.load_weights("results/a3/unet_model_weights_checkpoint.h5")
    # unet.train_unet(greys_imgs, labels_imgs, model)
    predict_unet(greys, model, saveDir)


def predict_unet(greys, model, savedir="./"):
    images = [d.imread(img) for img in greys]
    for name, img in zip(greys, images):
        res = unet.predict_single_image(model, img, batch_size=4)
        # print("There are {} nans!".format(np.count_nonzero(~np.isnan(res))))
        path, base, ext =  util.path_base_ext(name)
        d.imsave(savedir + "/" + base + '_predict' + ext, res.astype('float32'))


if __name__ == '__main__':
    main(sys.argv[1])

