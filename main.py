import sys
# sys.path.append("./models/")
import unet
from skimage.io import imread
import datasets as d
import util

def train(savedir):
    greys = util.sglob("data3/labeled_data_cellseg/greyscales/down6x/*.tif")
    greys_imgs = [d.imread(img) for img in greys]
    labels = util.sglob("data3/labeled_data_cellseg/labels/down6x/*.tif")
    labels_imgs = [d.imread(img) for img in labels]

    # labels_imgs = [img[0] for img in labels_imgs]
    for img in labels_imgs:
        img[img!=0]=2
        img[img==0]=1
        img[img==2]=0

    print("Input greyscale images:")
    for name,img in zip(greys, greys_imgs):
        print(name, img.shape)
    print("Input label images:")
    for name,img in zip(labels, labels_imgs):
        print(name, img.shape)

    unet.savedir = savedir
    unet.x_width = 120
    unet.y_width = 120
    unet.step = 4
    unet.batch_size = 32
    unet.learning_rate = 0.0005
    unet.nb_epoch = 300
    model = unet.get_unet()
    predict_unet(greys, model, savedir)

def predict(savedir):
    greys = util.sglob("data3/labeled_data_cellseg/greyscales/down6x/*.tif")
    greys_imgs = [d.imread(img) for img in greys]
    model = unet.get_unet()
    unet.savedir = savedir
    unet.x_width = 120
    unet.y_width = 120
    unet.step = 30
    model.load_weights("results2/b3/unet_model_weights_checkpoint.h5")

    print("Input greyscale images:")
    for name,img in zip(greys, greys_imgs):
        print(name, img.shape)

    for name, img in zip(greys, greys_imgs):
        res = unet.predict_single_image(model, img, batch_size=30)
        # print("There are {} nans!".format(np.count_nonzero(~np.isnan(res))))
        path, base, ext =  util.path_base_ext(name)
        d.imsave(savedir + "/" + base + '_predict' + ext, res.astype('float32'))
    predict_unet(greys, model, savedir)

if __name__ == '__main__':
    predict(sys.argv[1])

