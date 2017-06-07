import sys
# sys.path.append("./models/")
import unet
from skimage.io import imread
import datasets as d
import util




def train(savedir):
    all_grey_names = []
    all_grey_imgs = []
    all_label_imgs = []

    # build training dataset with all available training data
    grey_names = util.sglob("data3/labeled_data_cellseg/greyscales/*.tif")
    label_names = util.sglob("data3/labeled_data_cellseg/labels/*.tif")
    grey_imgs = [d.imread(img) for img in grey_names]
    label_imgs = [d.imread(img) for img in label_names]
    label_imgs = [img[0] for img in label_imgs]
    for img in label_imgs:
        img[img!=0]=2
        img[img==0]=1
        img[img==2]=0
    all_grey_names += grey_names
    all_grey_imgs += grey_imgs
    all_label_imgs += label_imgs

    grey_names = util.sglob("data3/labeled_data_cellseg/greyscales/down3x/*.tif")
    label_names = util.sglob("data3/labeled_data_cellseg/labels/down3x/*.tif")
    grey_imgs = [d.imread(img) for img in grey_names]
    label_imgs = [d.imread(img) for img in label_names]
    #label_imgs = [img[0] for img in label_imgs]
    for img in label_imgs:
        img[img!=0]=2
        img[img==0]=1
        img[img==2]=0
    all_grey_names += grey_names
    all_grey_imgs += grey_imgs
    all_label_imgs += label_imgs

    grey_names = util.sglob("data3/labeled_data_cellseg/greyscales/down6x/*.tif")
    label_names = util.sglob("data3/labeled_data_cellseg/labels/down6x/*.tif")
    grey_imgs = [d.imread(img) for img in grey_names]
    label_imgs = [d.imread(img) for img in label_names]
    #label_imgs = [img[0] for img in label_imgs]
    for img in label_imgs:
        img[img!=0]=2
        img[img==0]=1
        img[img==2]=0
    all_grey_names += grey_names
    all_grey_imgs += grey_imgs
    all_label_imgs += label_imgs

    print("Input greyscale and label images:")
    for n,g,l in zip(all_grey_names, all_grey_imgs, all_label_imgs):
        print(n,g.shape, l.shape)

    # valid training and prediction params (change these before prediction!)
    unet.savedir = savedir
    unet.x_width = 120
    unet.y_width = 120
    unet.step = 30

    # just training params
    unet.batch_size = 32
    unet.learning_rate = 0.0005
    unet.nb_epoch = 100
    unet.samples_per_epoch = 4000

    model = unet.get_unet()
    #model.load_weights("results2/b3/unet_model_weights_checkpoint.h5")
    unet.train_unet(all_grey_imgs, all_label_imgs, model)
    predict(savedir, all_grey_names, model_in_savedir=True)

def predict(savedir, greys=None, model_in_savedir=False):
    if greys is None:
        #greys = util.sglob("data3/labeled_data_membranes/images_big/smaller6x/*.tif")
        #greys = util.sglob("data3/labeled_data_membranes/images_big/*.tif")
        #greys = util.sglob("data3/labeled_data_membranes/images/*.tif")
        #greys = util.sglob("data3/labeled_data_membranes/images/small3x/*.tif")
        greys = util.sglob("data3/labeled_data_cellseg/greyscales/down6x/*.tif")
    
    grey_imgs = [d.imread(img) for img in greys]
    unet.savedir = savedir
    unet.x_width = 120
    unet.y_width = 120
    unet.step = 10
    model = unet.get_unet()

    if model_in_savedir:
        model.load_weights(savedir + "/unet_model_weights_checkpoint.h5")
    else:
        model.load_weights("results4/seg_down6x_2/unet_model_weights_checkpoint.h5")
#
    print("Input greyscale images:")
    for name,img in zip(greys, grey_imgs):
        print(name, img.shape)

    for name, img in zip(greys, grey_imgs):
        res = unet.predict_single_image(model, img, batch_size=30)
        # print("There are {} nans!".format(np.count_nonzero(~np.isnan(res))))
        path, base, ext =  util.path_base_ext(name)
        d.imsave(savedir + "/" + base + '_predict' + ext, res.astype('float32'))

if __name__ == '__main__':
    train(sys.argv[1])

