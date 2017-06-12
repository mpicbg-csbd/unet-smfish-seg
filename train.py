import sys
import unet
from skimage.io import imread
import datasets as d
import util


train_params = {
 'savedir' : './',
 'grey_tif_folder' : "data3/labeled_data_cellseg/greyscales/down6x/",
 'label_tif_folder' : "data3/labeled_data_cellseg/labels/down6x/",
 'x_width' : 120,
 'y_width' : 120,
 'step' : 30,
 'batch_size' : 32,
 'learning_rate' : 0.0005,
 'nb_epoch' : 100,
 'samples_per_epoch' : 'auto'
}




def train(train_params):
    train_grey_names = []
    train_grey_imgs = []
    train_label_imgs = []

    # build training dataset with all available training data
    #    grey_names = util.sglob("data3/labeled_data_cellseg/greyscales/*.tif")
    #    label_names = util.sglob("data3/labeled_data_cellseg/labels/*.tif")
    #    grey_imgs = [d.imread(img) for img in grey_names]
    #    label_imgs = [d.imread(img) for img in label_names]
    #    label_imgs = [img[0] for img in label_imgs]
    #    for img in label_imgs:
    #        img[img!=0]=2
    #        img[img==0]=1
    #        img[img==2]=0
    #    train_grey_names += grey_names
    #    train_grey_imgs += grey_imgs
    #    train_label_imgs += label_imgs

    grey_names = util.sglob(train_params['grey_tif_folder'] + "*.tif")
    label_names = util.sglob(train_params['label_tif_folder'] + "*.tif")
    grey_imgs = [d.imread(img) for img in grey_names]
    label_imgs = [d.imread(img) for img in label_names]
    for img in label_imgs:
        img[img!=0]=2
        img[img==0]=1
        img[img==2]=0
    train_grey_names += grey_names
    train_grey_imgs += grey_imgs
    train_label_imgs += label_imgs

    # grey_names = util.sglob("data3/labeled_data_cellseg/greyscales/down6x/*.tif")
    # label_names = util.sglob("data3/labeled_data_cellseg/labels/down6x/*.tif")
    # grey_imgs = [d.imread(img) for img in grey_names]
    # label_imgs = [d.imread(img) for img in label_names]
    # #label_imgs = [img[0] for img in label_imgs]
    # for img in label_imgs:
    #     img[img!=0]=2
    #     img[img==0]=1
    #     img[img==2]=0
    # train_grey_names += grey_names
    # train_grey_imgs += grey_imgs
    # train_label_imgs += label_imgs

    print("Input greyscale and label images:")
    for n,g,l in zip(train_grey_names, train_grey_imgs, train_label_imgs):
        print(n,g.shape, l.shape)

    # valid training and prediction params (change these before prediction!)
    unet.savedir = train_params['savedir']
    unet.x_width = train_params['x_width']
    unet.y_width = train_params['y_width']
    unet.step = train_params['step']

    # just training params
    unet.batch_size = train_params['batch_size']
    unet.learning_rate = train_params['learning_rate']
    unet.nb_epoch = train_params['nb_epoch']
    unet.samples_per_epoch = train_params['samples_per_epoch']

    model = unet.get_unet()
    # model.load_weights("results2/b3/unet_model_weights_checkpoint.h5")
    unet.train_unet(train_grey_imgs, train_label_imgs, model)


if __name__ == '__main__':
    train_params['savedir'] = sys.argv[1]
    train(train_params)
