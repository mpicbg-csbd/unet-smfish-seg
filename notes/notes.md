Here's my current list of problems.

*What library and tool to use for learning?*

- [ ] We either have to convince Ulrich to merge Dave's changes (not going to happen), or we have to replace Cascaded Random Forests with something else... (possible?) or we have to reimplement CRF in a way that doesn't force us to fork vigra...

1. What are the end results. Removing Dave's changes to vigra, and relying on vigra at a lower level would remove some complexity. Then we can use cmake to build just dave's stuff in a cross platform way. But your average user still has to install vigra! (and boost? and hdf5? and the right version of gcc? etc...)

2. Or, we see if it's possible to move all of the training and prediction into python *without experiencing a slowdown*.

3. Or, we use neural nets?

*How to build on my Mac?*

Assuming that we still use Random Forests...

- [ ] build dave's vigra ~~or install a replacement~~
- [x] install openGM, potentially HDF5 and boost
- [x] format the cxx code
- [ ] submit a pull request

*Port feature creation from Java*

This doesn't depend on whether we use dave/python/ for learning. As long as we are using Random Forests at some level, then we want to make features.

- [ ] move feature creation from KNIME/Java into vigra. Use c++ (and leave dave's code alone) or use python, and move dave's CRF into python as well!

*How to distribute?*

Vigra is installed on Mac via brew. Linux has their own package managers. How do we install vigra on windows? I have no idea. Maybe it comes with binary installers...

*How to do software testing?*

See "KNIME training data / software testing data"


# Keras & Deep learning for pixelwise classification

- Any reason to prefer RF for pixelwise classification?
- Good examples / network architectures for pixelwise classification / regression?

*We don't want a classifier! We want something that returns a score/probability/continuous value that predicts membrane / branching-point pixels*

- deconvolution layers
  Do they actually perform deconv? Isn't that usually an iterative & expensive computation?

# Dagmar's training data

If we want to train a new classifier in the future we want to use all the data available in the `labeled_data` folder, but watch out! they come in two different resolutions from the two different objectives used.

The stuff in `./labeled_data/20151016 Ground truths for 100x objective/` has much larger looking cells, in addition to have much larger images. But the biology in the two is the same, so with appropriate downscaling we should be able to mush them all together into on big training set[^1].

[^1] or more appropriately Training + Validation + Testing set...

We have data in many different sizes...
Carine's original data in `resources/testing_data/used_for_training_by_dagmar/all/grayscale` is 3147 x 1423
The cropped versions at same resolution in the `Cropped` subfolder are 2128 x 932
The training data used in `resources/testing_data/used_for_training_by_dagmar/fold*` are full resolution and size: 3147 x 1423.
`resources/20151016 Ground truths for 100x objective` contains membrane ground truth in very large & high-res images 9600 x 4000

The cropped images have been moved into `test_data` and separated into training and testing batches. (maybe scikit learn will do this for us automatically?)


# KNIME training data / software testing data

`knime_test_data/` contains the original training and example prediction data from the currently used KNIME workflow. I've arranged it into training and testing components, each of which contain an input and output. These are all images, with the exception of the output for training, which is a random forest binary blob, `rfcascade`. The input to `predict` must include `rfcascade`, as well as all the features created in KNIME...

# Tue Jan 24

Going to download / link to Carine's ground truth for membrane & branching points.

> You can find the ground truth data on the ‘zebrafish-membranes’ project space. The ground truths that we used for training the pipeline that is online are in the folder ‘20150702 Ground truths’ I believe.
There are additional ground truths in the folder ‘20151016 Ground truths for 100x objective’ in case you have a use for those ;).

Thanks Carine!

# Wed Jan 25

Plan for today. I can use Carine's ground truth to train a classifier. *Without* Cascading! Just with a big pile of features and vigra's standard RFs. One insane thing is that we never did a thorough comparison of the Cascaded RF with the Weka RF, when both were using the Weka features! We only saw that dagmar's cascaded RF with Amira features were much better than Weka RF with Weka features... But the current classifier results don't look *nearly* as good as I remember her Amira + CRF output looking... *Can we find her original output and use it as a comparison?*
*Does Amira roll it's own feature creation? Do they use vigra?*

It would be good today to try training a RF using ilastik to do the membrane and vertex prob maps. One alternative is to try training a NN to create effective probability maps.

- [ ] Ask Laurent & Uwe about a good architecture for this. I've got 1M pix / image * 10 images = 10M pix. 1/20 of that is membrane class. 1/200 is vertex. 500k membrane, 50k vertex.

- [ ] TODO in a perfect world we would automatically downscale images to be the right size (before running prediction) by identifying the content inside the image...

# Thu Jan 26

I need to make the simplest possible three-class learner using lots of features from scikit learn? Let's try to do this in ilastik...

When I do the naive thing and just try running all the functions available in skimage.filter on a 2d float image of Carine's tissue I get all kinds of errors. The most common one says: "Images of type float must be between -1 and 1." These are ValueErrors that can probably be easily fixed...

If we start off with an image already in [-1, 1] then we get many more successful images written to disk, but looking through most of them shows that they are just pure black...

That's because you were saving them as uint16's ! If you save them as float32's then they look great. They are all saved as float32s and they re-open as float32's. In Fiji they import correctly as 32bit images with pixel values in the correct range. [-1, 1]. i.e. Fiji also allows negative values for images.

*What about the remaining errors?*

Well for `threshold_li` I get an AssertionError, which means they caught a worse error early... This is because it doesn't return an image, like the rest of them do, but rather it just returns a number (the threshold). This is a little bit obnoxious from an API point of view...

Note that one bad error is that the threshold functions actually spit out a broken image, despite being caught by assert!

F.`deprecated` is just a function decorator picked up by the module's dictionary...
F.`copy_func` does something I don't really understand... copy a function?

Some are just old, deprecated names, e.g. `vsobel`...

One serious error is:
`gabor() takes at least 2 arguments (1 given)`

# Mon Jan 30

I've got a working Random Forest from scikit-learn! It can spit out class predictions and probability maps. Next I'd like to use the `xgboost` trees to see how they do compared to the cascaded RFs. Also, what is *auto-context*, and how does it fit in with these ideas? Does the cascaded RF that Dave uses work with auto-context? *What if we just use the original greyscale image but shifted in x&y by a few pixels?* [auto-context]

Tried to start a spec for the data/directory structure that we want before feeding it into the pipeline. The nice thing about a spec, as opposed to just adding new code to read the existing data, is that when you distribute your program it is easier for users to conform their data to your spec then for you to write / add new code to read their data!!!

[auto-context]: http://pages.ucsd.edu/~ztu/publication/cvpr08_autocontext.pdf

# Tue Jan 31

Implemented subsampling in class-specific ways. We can choose the number of datapoints that we want to train on from each class.

NOTE: It's very important, when judging the quality of your output by eye, do adjust the brighness threshold s.t. only the top 5-10% of probabilities show up as non-black! This makes our really shitty results all of the sudden look really good!

In `knime_test_data/data/predict/Results/PredictKNIME`:

The output `predict_20150127_EVLvsInner01_slice11-normalized_level1_smooth0.tif` is just pure black! And all the images with "smooth" appended to the name have been blurred too much. How can we compare the classification results of the CRF to those of sklearn?

Also, `log.txt` is totally blank.

Another good reason for going with python standards... Many more devs (of the appropriate kind) can work with this code than can work with Java and ImgLib. We want data scientists, not Java GUI interface developers.

Also makes it much easier to distribute you code & run on cluster.

First, let's try to restrict the number of features used per tree to the recommended 20. Now it's starting to take a few seconds to train... ?why does adding an upper limit to the feature number make things slower? aren't there fewer features to search through?

Random Forest param search.
[https://hal.archives-ouvertes.fr/hal-00436372/document]
<= 20 random features / tree.
numpy alternative. [https://github.com/dmlc/minpy]

----

I'm struggling with how to write things. How and when to open and save my data. Do I want to open large things once at the beginning and have them sit around in an object until the whole program is done? Or only open them within the function in which they are needed, then close them again at the end. Will this force my functions to be large? Do I want to use classes/objects or just dictionaries? Do I want to put complex objects inside my dictionaries or not?

TODO:
*The filters used on images allows us to do a pixel-wise classification of images, but there are also more coarse-grained features we might want to use... Why not build a global forest which includes the output from skimage.feature as well as filters?*
*How would we incorporate these global properties into our pixel-patch classification decisions in a smart way?*

# Wed Feb 1

Branching project! Now we're separating the project into two pieces. One that depends on Dave's CRF code, and one that only uses scikit/numpy/keras stuff, see `carine_skit_RF`...

----------

From here on down we're in the scikit-learn branch of the project!

# Thu Feb 2

I want to do two more things with this project.

1. Get some segmentations based off of the membrane+vertex prob maps.
2. Try to use keras to create those prob maps.

# Fri Feb 3

Got some basic output from my keras model, with the help of Laurent. It doesn't seem to be able to learn the difference between the classes, except for predicting inside the tissue as membrane/vertex and outside the tissue as background... And we've tried many different reweightings of the classes, as well as adjusting the input images to set the background to zero.

The next most useful thing you can do is make new featurestacks from scikit-image, and use your existing code to actually do the segmentations! (Run it as an uberjar?)

Laurent's suggestions for improving the keras model are:

1. remove the pure background from the samples
2. increase the data amount by reducing the sampling stride
3. adjust the model


Maybe try using these new input images with zero background

# Mon Feb 6

Writing some notes on why the KNIME project has to be replaced here : `./KNIME_carine_post-mortem.txt`.

I've got a new list of things to do in my notebook. One of the first I need to do is get full segmentations working on the results of my random forests here.

I can just use Fiji as it currently exists!

Laurent advice. Merge labels into 2 classes.

Try U-net and membrane classifier from Thorsten that won CREMI challenge?

Add to the list of obnoxious things about our KNIME workflow: The quality metrics you would expect to see after training the CRF were all hidden!

---

Tried to run the Cascaded Random Forest using the bash scripts and KNIME directory structure, but ran into an error "Precondition Violation... " then I fixed it by removing a file from the directory that wasn't numbered, then I got a silent failure after reading "initialization succeeded." There was simply no image output in the Results directory...

Moved the Random Forest into what I hope was the correct directory and BINGO I got some additional output... It worked! Now I'll make a comparison between the CRF and xgboost.

# Tue Feb 7

I have a better understanding now of why Dave's CRF wont work on linux. The `learn` binary requires a very special commit from Dave's code. We don't know which one, because we never got the source, only the results of the build. We could try building on every commit, and running tests until we find one that passes the tests... But it will very likely be the wrong commit...

RESULT 1! from crf_vs_xgboost.py
```
In [41]: vs.run_jac()

./knime_test_data/data/train/PredictKNIME/grayscale_0_level1_probs1.tif
    CRF class1: 	 0.9744 	   0.5
xgboost class1: 	 0.9735 	  0.75
    CRF class2: 	 0.9651 	 0.333
xgboost class2: 	 0.9648 	 0.417

./knime_test_data/data/train/PredictKNIME/grayscale_1_level1_probs1.tif
    CRF class1: 	 0.9788 	 0.417
xgboost class1: 	 0.9789 	  0.75
    CRF class2: 	 0.9751 	 0.375
xgboost class2: 	 0.9752 	 0.458

./knime_test_data/data/train/PredictKNIME/grayscale_2_level1_probs1.tif
    CRF class1: 	 0.9789 	 0.375
xgboost class1: 	 0.9782 	  0.75
    CRF class2: 	 0.9747 	  0.25
xgboost class2: 	 0.9743 	 0.458

./knime_test_data/data/train/PredictKNIME/grayscale_3_level1_probs1.tif
    CRF class1: 	 0.9787 	 0.375
xgboost class1: 	 0.9783 	 0.708
    CRF class2: 	 0.9733 	  0.25
xgboost class2: 	 0.9731 	 0.375
```

This result means that the xgboost predictor is about as accurate as the CRF. But it also appears to find it's maximum accuracy at a more consistent value of the threshold! This means we can fix the threshold in the cell-segmentation pipeline without worrying we have a bad segmentation!

Laurent's advice... Build the confusion matrix for classification. Do the evaluation on all the possible metrics (or at least a lot of them) and if they all sorta agree on a winner then it's pretty clear. Use the jaccard index for vertex class, but use l1 or l2 norm for membrane, because we never do a flat segmentation on it during the cell segmentation.

# Wed Feb 8

Try out TensorBoard callback function for monitoring training in keras.

Our pixelwise-classifier ground truth is not a valid cell segmentation. So we can't use it for evaluating our pipeline... But the ground truth i just found on Carine's server is actually useable...

Also, in general it would be great if pixelwise GT labels, colors, masks, etc were all just different channels in the same tif image, and all in the same folder. This would would reduce complexity. Also, channels should probably be named... Not just indexed with a number like 0,1,2,3... But rather "red" "green" "membrane-label" "cell-label" etc...

The problem with making them different channels is they have to be the same dtype... and labels would rather be uints and greyscale / intensity info would rather be floats...

but mushing them into one image avoids having to keep the names consistent between different folders... And means all your cropping / scaling works across channels as well. (and will warn you that continuous transformations of your intensity image will not map nicely onto your label image!)

It's most helpful to look at the confusion matrix! It shows us how the different classifiers behave very differently! xgbs really likes to make the membranes large, and favors

RESULT 1.1 !
```
In [177]: crfs_confusion
Out[177]:
array([[838268,   3181,    196],
       [ 11065,   8627,    245],
       [  2839,   1366,    597]])

# 2nd version where we assume vertex prob is add to membrane prob during classification.

array([[840204,   1441,      0],
      [ 14101,   5836,      0],
      [  4400,    402,      0]])

In [178]: xgbs_confusion
Out[178]:
array([[809777,  31382,    486],
       [  1204,  18603,    130],
       [   326,   3715,    761]])
```
This is the sum total across four test images. The first dimension = outer list = row corresponds to the ground truth class, while the column corresponds to the predicted class. {0:background 1:membrane 2:vertex}. We can see that the gradient boosted trees are more accurate on membrane and vertex, but less on background.

TODO: more test images.
TODO: test cell segmentations!
TODO: keras classifier!

TODO: In the xgbs model a pixel is either membrane OR vertex, while in the CRF model a pixel can be both membrane AND vertex. Which makes more sense? If you model the classes as mutually exclusive then later in the cell segmentation you have to merge them back together for creating the paths! But what do we do if a pixel is membrane in the membrane classifier but background in the vertex classifier? Then it's membrane. What about vertex in the vertex classifier, but background in the membrane classifier??? This is trickier... You probably always want to air on the side of membrane/vertex, because missing a vertex is a harder problem for the segmenter to fix than having an extra vertex.

# Fri Feb 10

I wanna make a smart featurestack from skimage features and filters. How do I decide on the best param ranges? You don't need best, you just need reasonable. Try out a selection of features for a couple different parameter ranges?

Trying out a range of Gabor filters for different wavelengths. For very long wavelengths the filters are very slow to compute...

Trying out retraining with the class_weights='balanced' option. The results are *much* better when predicting on the training data.

Wanna see how the number of training samples affects performance?

```

Here we only train on a random subset of 3000 pixels.

In [4]: rafo = rf.train_rafo_from_stack(wekastack, lab)
   ...: ypred = rf.predict_from_stack(rafo, wekastack)
   ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
   ...:
[[211150      1      0]
 [  4517     43      0]
 [   875      0     10]]

In [5]: rafo = rf.train_rafo_from_stack(wekastack, lab)
   ...: ypred = rf.predict_from_stack(rafo, wekastack)
   ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
   ...:
[[167235      0  43916]
 [  3470     32   1058]
 [   682      0    203]]

In [6]: rafo = rf.train_rafo_from_stack(wekastack, lab)
   ...: ypred = rf.predict_from_stack(rafo, wekastack)
   ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
   ...:
[[167381  43768      2]
 [  3459   1101      0]
 [   673    201     11]]

Here we train on a small subset, the first 3000 pixels in the array.

In [7]: rafo = rf.train_rafo_from_stack(wekastack, lab)
   ...: ypred = rf.predict_from_stack(rafo, wekastack)
   ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
   ...:
[[167354      1  43796]
 [  3462     40   1058]
 [   677      0    208]]

In [8]: rafo = rf.train_rafo_from_stack(wekastack, lab)
   ...: ypred = rf.predict_from_stack(rafo, wekastack)
   ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
   ...:
[[167339      5  43807]
 [  3441     61   1058]
 [   674      0    211]]

 Here we ugrade from training on a subsample of 3000 pixels to the full image.

In [9]: rafo = rf.train_rafo_from_stack(wekastack, lab)
   ...: ypred = rf.predict_from_stack(rafo, wekastack)
   ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
   ...:
[[167261  43885      5]
 [   389   4171      0]
 [    22    202    661]]

In [10]: rafo = rf.train_rafo_from_stack(wekastack, lab)
    ...: ypred = rf.predict_from_stack(rafo, wekastack)
    ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
    ...:
[[167280  43866      5]
 [   420   4140      0]
 [    24    201    660]]

In [11]: rafo = rf.train_rafo_from_stack(wekastack, lab)
    ...: ypred = rf.predict_from_stack(rafo, wekastack)
    ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
    ...:
[[167278  43868      5]
 [   336   4224      0]
 [    16    201    668]]

Here we use Gabor featurestacks...

In [12]: rafo = rf.train_rafo_from_stack(gaborstack, lab)
    ...: ypred = rf.predict_from_stack(rafo, gaborstack)
    ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
    ...:
[[157142    671  53338]
 [    92   3334   1134]
 [     6      0    879]]

In [13]: rafo = rf.train_rafo_from_stack(gaborstack, lab)
    ...: ypred = rf.predict_from_stack(rafo, gaborstack)
    ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
    ...:
[[157354    462  53335]
 [    87   3338   1135]
 [    12      0    873]]

In [14]: rafo = rf.train_rafo_from_stack(gaborstack, lab)
    ...: ypred = rf.predict_from_stack(rafo, gaborstack)
    ...: print(sklearn.metrics.confusion_matrix(lab.flatten(),ypred))
    ...:
[[157540    279  53332]
 [    80   3345   1135]
 [     7      0    878]]
```

# Wed Feb 15

I'm having some troubles keeping things straight that I think would be benefitted by a type system. I want to be able to train a random forest on a single image, with a single featurestack, with subsets of the training data, on a list of images, on a concatenated numpy array of images, or on a directory. That's a lot of potentially differently-typed input! Some of these different things can be run through the same code without any trouble, for example a single featurestack can probably use the same code as a concatenated set of featurestacks...

Can we solve this problem by labeling the axes of our arrays?

Do we need different code for objects of the same type? i.e. featurestacks and concatenated featurestacks?

ERRATA 1.1:
We don't know how the CRFs were trained, but probably both the CRFs and the XGBOOST trees were trained on all four labeled images. Which means we're only testing model power (ability to overfit) in RESULT 1.1 and not ability to generalize.

Accidentally shuffled both X's and Y's in totally unrelated ways. Then tried predicting. Got exactly zero membrane and vertices correct on testing data, but still got a bunch of background correct. Also, got a lot of both correct on the training data! This would be a good test to do every time you train a classifier...

Fixing the shuffling issue brings the training and testing data much closer together. This shows us how important spatial correlations are!

Training an XGBOOST classifier without balancing the data results in both Train and Test only classified as background!

# Thu Feb 16

I've got enough files now and enough different experiments where I need to break things up a bit more. I'm going to make a scripts directory for standalone scripts, i.e. things that take care of their own reading/writing/importing files and modules.

Maybe there should also be some common utility functions which we can separate into a `common` or `util`. But the bulk of our code is not standalone scripts, but rather functions which work with python data and don't need to know about the filesystem or global state at all.

Now I've got a bunch of stuff in `keras_classifier.py` that talks to the filesystem. I'm going to remove that bit and only have it talk to `imglist`s, which are lists of (potentially differently sized) ndarrays corresponding to images. We just need function which turn these image lists into the shapes of things that go into the learers. For random forests, X.ndim = 2, but the size will vary for many keras models, including (n_patches,xdim,ydim,n_channels) image patch arrays. Since X is model specific, each model should have it's own imglist -> X function. *what about preprocessing?* yes, each model should have it's own preprocessing, becuase that's really a part of the model and something we will do a lot of experiments with... But if we can have a Model class with a `preprocessing` and `fit` and `imglist->X` and `predict`... that would go a long way to enabling code that works with RF and keras... but they are so different we might just want use separate interfaces anyways...

This also means that each model should have it's own module, and that we should have a `models` directory for storing them...

## Pros and cons of storing the preprocessing with the model

Pro
---
1. They are always applied in pairs. Preprocessing only makes sense if followed by a model and vice versa. In many ways the preprocessing is really a part of the modeling itself.
2. You won't forget which type of preprocessing you've done, since they are always coupled in the same file.
3. Much of the preprocessing will be on full images, i.e. before we translate the imglist into `X` and `Y`, so they can be shared.

Con
---
1. But We might want to mix and match preprocessing approaches with modeling approaches, because they are also largely independent. If we have the same type of X [same shape] then we can apply the same preprocessing routines on them, and that code should live in a shared library. But to the extent that these things are independent and unshareable they should live with the modeling code.

----------

*Do we even want to preprocess our images?*
I thought preprocessing just destroyed potentially useful/informative variance? Won't our learners just learn to deal with the data the way it is?

*What about data augmentation?*
This will likely be required by some data-hungry models. This can/should also be separate, so that we can also try feeding the augmented data into the RandomForest models just for fun?

What responsibilities fall within the `imglist_to_XY` function vs the `preprocess` function? At which point do we make sure that we have the correct types, etc? Since the types involved are closely related to the normalization of the data it makes sense to keep all the type conversion ONLY in the preprocessing step.

# DAWN OF THE DEEP U-NETS
=========================

I've got my first Unet trining here and the accuracy on the validation set > 1k samples is already .9668 and growing. I don't know if we ever had accuracy this good with the Random Forests? But we still don't know if this is the same local minimum that we get just by classifying everything as background.

# Fri Feb 17

Let's try running the net on a piece of image with both classes.

# Sat Feb 18

First, let's write a function that will run our unets on an imagelist and write out predictions which are full images, sown together of little patches. Then we'll combine all the preprocessing into the `imglist_to_XY` function.

NOTE: Martin's good point. The model we've trained is only a couple of convolutions and then upsampling, each of which change your patch-size by a constant factor, but don't depend on the input patch size being fixed! We could build a different model with a different Input/Output shape & larger patches, but then just use the weights from this model without retraining. *Our convolutions are still being applied to pixel patches 2x2 and 3x3*, so it's not like it messes up the scaling. It's just that we have to apply many more convolutions at each layer. This number is implicitly defined in the model = img-width*img-height/2x2 or 3x3.

TODO: Add an Image object which extends ndArray in the same way as Julia's Image, so that images can have metadata that they carry along, without getting decoupled/lost when you're shuffling data around.

# Sun Feb 19

8386s per epoch with "Train on 103101 samples, validate on 25776 samples".

It's hard to get exactly the right api. A problem I have at the moment is some functions work at the dataset level (lists of images, full X,Y vectors, etc) and some work at the single image level. X and Y mush image data together, duplicate image data, and can't be pieced back together to make images at all. Maybe we don't want them to be? But probably we do. This requires saving lots of extra info (coordinates for each patch as well as which image those coordinates belong to, as well as keeping the coordinate vector in sync with the X and Y vectors). The way we would do things with Random Forests was to split data up into X/Y_train and X/Y_test. We would calculate our metrics based on the test data, which was fine because each sample in X&Y was an individual pixel, only seen once in the true images. (Although sometimes we would only evaluate a subsample?) But with patch-based classifiers we have duplicated data, so now the accuracy across patches doesn't reflect the accuracy across our images...... No I'm not sure there's a difference, because we still have to break our images up into overlapping patches... In the end we need our images to correspond to classifications, so we must be able to combine the patches in a smart way... ~~TODO: when averaging overlapping patches together this should be done BEFORE we apply the argmax to turn our class-scores into a firm class decision...~~

We want to be running code on the cluster, on full datasets, not interactively in ipython, on single images!

RESULT 1.2
```
In [6]: ipy.train_and_test_rafo_gabor()
Class Dist:  (array([0, 1, 2], dtype=uint8), array([673193,  16051,   3863]))
confusion_matrix Train:
[[177552 313178 182463]
 [  3294   8740   4017]
 [   544   1441   1878]]
confusion_matrix Test:
[[43544 79089 45819]
 [ 1058  1762  1066]
 [  248   438   253]]
```
Gabors by themselves are terrible at predicting! How can this even be possible?

Test: try kmeans, or a "decision tree" just based off of the intensity feature alone.

I've given my model big patches to look at and I've used stride 5. I'm letting it train for 40 mins, but the loss and accuracy just don't change at all.

# Mon Feb 20

Now that I've got a system for training, predicting, and retraining my models I can continually improve on a single model until it's optimal. But this means I can change the learning rate as i see the model have a harder and harder time decreasing with each step.

Setting up my Ubuntu 12.4.3 machine required...

```
sudo apt-get install git
sudo apt-get install python-dev  # for python2.x installs
download get-pip.py and install
pip install --user numpy scipy ipython nose scikit-learn scikit-image
pip install --user keras
sudo apt-get install python-tk

# necessary for xgboost
sudo apt-get install g++
build and install xgboost (see their site. git clone + make -j4)
setup xgboost python package installation
pip install --user tensorflow
pip install --user tensorflow-gpu
```

BUT in the end, tensorflow was not compatible with Ubuntu 10.x, so I asked Juraj to help upgrade to 12.x...

GPUs on myers-pc-2
```
broaddus@myers-pc-2:~/Desktop/Projects/carine_smFISH_seg$ lspci | grep VGA
05:00.0 VGA compatible controller: NVIDIA Corporation GF100GL [Quadro 4000] (rev a3)
22:00.0 VGA compatible controller: NVIDIA Corporation GF100GL [Quadro 4000] (rev a3)
```

It's important to appreciate that with machine-learned models with enough training data you get systems which are as accurate as *experts* in biological image analysis. Not just random people you hired off of amazon Turk. But people who spend their time looking at (e.g. fluorescence microscopy) images!

Thoughts after 14x5 epochs of training... We still haven't converged, although the learning rate has been decreased heavily. Now we're

Now let's make a really dumb segmentation algorithm... It will first apply a threshold, then id small connected components as cells, then it will do a seeded watershed at the center of each component on the membrane-activation channel.

NOTE: membrane activation channel + background channel not= 1! How should we deal with regions that are uncertain? What does it mean when both scores are low? high?

So many things are weird... why do my images look like they're filled with nans? the nans are still there after saving and importing. But the images look fine!

WE EXPECT BLACK BARS ON OUR PREDICTION IMAGES. THE BLACK BARS ARE NAN. They come from the pixels that never made it into any patch. When windowing, if your stride is large, you will miss pixels on the bottom and right borders, depending on how evenly your width/height is divisible by the step size and patch width.

TODO: move unseen data and labeled_data over to other machines. Setup synchronization with rsync.

# 02/21/17

*How well does the unet generalize to unseen images?*
*Is it OK to train the net on 80% of the data across all images, and test on the other 20%?*

*If we train on the unseen images only, how do the predictions change?* The unseen images were used in the paper and have labeled cells (from which we can infer the membrane labeling, and test ground truth segmentation...)

TODO: fix rotation problem... nevermind it's just that Carine rotated some of the images for the paper...

TODO: Image viewing problems...
How are we supposed to view greyscale/intensity images alongside label images? If we want to perge them into a single image we have to convert the types to be the same, which means embedding the uints inside the floats. Really we don't want to mush them into the same image, but just want them to zoom/pan/rotate together as one image, but we want the types and colorschemes to be different. We might even want to group a label image with an RGB image! Then we can view multpile intensity channels together for e.g. nuclei and membrane, blended, and then jump back and forth between the labels and the intensity.

ERROR!
When downscaling a labeled image you have to be sure *not* to average the labels together! How to do it? Does interpolation oder = 0 do the trick? Let's see, but I doubt it, as we don't really have to interpolate when downscaling, but I'm not sure how it's defined. What we really want is MIN_POOLING, where we downscale by an integer factor and, within each window, we take the min. This will keep our 0-valued membrane boundaries intact! (actually it will grow them!? we can't do this... Even a 1px wide 0-valued membrane would grow in width according to the scaling factor, and since the rest of the image would be smaller it would really grow. NO IT WONT GROW THEM, BECAUSE THE DOWNSCALING WINDOWS WILL NOT OVERLAP.

Min-Pool downscaling works!
Min-Pool downscaling doesn't work!!! We lose some cells in the process! This is a shame... Any cell smaller than a few pixels across will be totally lost in this process! But we can still compare the results of the downscaled labeling with the segmentation of the downscaled image, even if a few small cells got swallowed up in the process...

Failed to install tensorflow-gpu on my mac. Even with the correct LD flags. Not sure why. Uninstalled and reinstalled normal tensorflow. Which broke my code... Strange bug "TypeError: Expected int32, got list containing Tensors of type '_Message' instead." Need to downgrade my TensorFlow, as this appears to be a bug.

Had to uninstall tensorflow1.0.0 and reinstall 0.12. Which required me to use sudo, otherwise it would break upon cleanup after pip install....

But now at least things are back to where they were yesterday....

# 02/22/17

My unet is giving beautiful results on the training data ;). But there are strangely-dark patches on the `unseen` test data. IE it's overfitting, which I can see from the prediction results, but can't see *during* the actual training because i reshuffle the X/Y data every time I rerun the tests, which means I'm essentially training on 100% of the data, not just 80%...

Martin helped me set up my machine for using my CUDA-enabled GPU. 

The SEG segmentation measure is implemented wrong... or maybe has some secret hidden constraints that prevents it from working for anyone but the implementer.

# Thu Feb 23

Still getting shitty weird dark patches in my predictions... 
- Is it due to normalization?

TODO:
- Deform error
- RAND error
- pixelwise error

image viewer as a function that prepares a list of images (interpreted as a stack) for viewing with the system viewer, i.e. a properly scaled RGB image.

TODO: DONE:
Every TODO above this line has been ported over to Workflowy todo list

# Fri Feb 24 14:28:06 2017

# Wed Mar  8 10:54:39 2017

We really want two different workflows for training and prediction. Predictions should specify the model file to use, but the predictions do not go in the same folder as the model file.

Models result from code
Predictions ⤆ code + input images

**What is my current problem?**
- My cell segmentations aren't perfect (but they are quite good?)
- There are some pieces of membrane where I don't know the correct classification!
- The project isn't ready for use by 3rd parties.
- I can't compare cell segmentations with full size results!
  + Rotate original images.
  + Scale up the cell predictions
- Do we want predictions with vertices?
- I don't know what learning rate is best? (This is not an issue if the membrane maps are good!)
  + With learning rate 0.005 we can't escape the initial param regime where everything is grey...

I want to know if there are labeled versions of the images in "Cell segmentations paper/"... They are all labeled... 

---

I want to automatically segment my images. To find the best threshold segmentation level and compare it against snake-based segmentation. How easy would it be to 

OK! Everything in data2/ is now rotated correctly! Problems solved.

Can't compare the cropped images with the cell segmentation GT!
Gotta predict on the non-cropped imgs...

FOUND THE REMAINING CELL SEGMENTATIONS!!!

Now we really have ALL the labeled data AND the full size input images.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                                                                     │
│             ____  ____  ____  ____  __    ________  ________        │
│            / __ \/ __ \/ __ \/ __ )/ /   / ____/  |/  / ___/        │
│           / /_/ / /_/ / / / / __  / /   / __/ / /|_/ /\__ \         │
│          / ____/ _, _/ /_/ / /_/ / /___/ /___/ /  / /___/ /         │
│         /_/   /_/ |_|\____/_____/_____/_____/_/  /_//____/          │
│                                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

PROBLEMS. IN CHRONOLOGICAL ORDER.

# PROBLEM: GPU memory explodes

You can fix this by figuring out the function (dataset,model) → (Max GPU memory usage, Max RAM Usage, Time taken). This shouldn't even be that hard to do!

# Thoughts: Thu Mar  9 13:28:55 2017 -- Early stopping, weak models & proper scaling

I submitted a job to the cluster last night: `results/bigim3/`. It was supposed to run for 300 epochs, but only ran for 18. There were no signs of error in the stderr log, or in stdout.

The callbacks I was running include...
checkpoint and earlystopper.

I bet it was earlystopper on the validation loss, which appeared not to be advancing very much any more.

I changed the verbosity to 1, so it will tell me when it causes an earlystop, but I aslo bet there was a problem that the model was just not powerful enough.

**Proper Scaling**
We can either try to fix this problem by adding more layers, or by decreasing the size of the input images. Our normal size is 6x smaller. This leads to good predictions, but we have trouble upscaling the images again.

If we're going to get good cell segmentations, we don't want to be upscaling images by a factor of 6!!! This will completely destroy many of the cells in the ground truth, turning them into things only a few pixels on a side...

We want to know how: *how much can we downscale the image and still get 100% correct cell segmentations after upscaling?* We can answer this question (and the maximum accuracy of our membrane segmentations) because we have access to cell and membrane segmentation ground truth.

# Thu Mar  9 13:41:14 2017 -- Fiji vs leiningen & maven

I want to make quick clojure scripts that can interact with the images I have open in fiji. Why not just always work with ndarrays and python, and **only use fiji for viewing and measuring**?

My problem is when I use clojure with leiningen, importing all the java/imageJ classes that I need, I don't have access to the full power of Fiji; just ImageJ.

You can run simple clojure scripts from the command line with `fiji script.clj` to execute them with all of the Fiji classes & plugins on the classpath! This is really powerful! But it is also totally non-standard. Would the REPL still work? No, because you are executing via a totally separate mechanism. But fiji has it's own REPL! But this would totally ignore `leiningen`. It might not even be able to pick up the other local clojure files, unless it automatically adds the local dir to the classpath. The behavior you get when running via fiji and via Lein or standard clojure will not be in general the same, because the package versions will be different between fiji and maven.

-------------------------------------------------------------------------------

# Thu Mar  9 16:16:44 2017 -- Analysis of ./results/halfhalf_pred4_2/

I've got my predictions back on the correctly-cropped images. They look similar but not identical to the predictions on the non-cropped training data. The membrane signal is strong and the connected-component cell segmentations are not too sensitive to the threshold level. The pixelwise accuracy on the test-data was 96.66%, but the cell segmentation accuracy was much worse:

Was able to match $col1 cells out of the $col2 cells in the ground truth and the $col3 cells in the predicted image.

matched  GT   predicted  Image
=======  ==   =========  =====
138      222  236        20150127_EVLvsInner01
139      247  267        20150128_fig10
187      364  384        20150206_older_stages_test02
77       128  175        20150211_mex3b_sox19a_domes03
139      241  236        20150211_mex3b_sox19a_domes04
136      177  247        20150211_mex3b_sox19a_domes09
161      246  273        20150211_mex3b_sox19a_domes10
159      245  300        20150211_mex3b_sox19a_domes19
113      256  190        20150215_fig3_sphere_repeat02
68       138  171        20150215_fig3_sphere_repeat07
117      195  265        20150215_fig3_sphere_repeat09
123      186  194        20150215_fig3_sphere_repeat11
137      204  259        20150215_fig3_sphere_repeat16
137      201  228        20150215_fig3_sphere_repeat17
79       200  275        20150430_eif4g_dome01
92       224  184        20150430_eif4g_dome03
107      189  268        20150430_eif4g_dome06
160      322  248        20150430_eif4g_dome07


*What are the problems?*

To identify the problems we probably want to look at the locations in the images which are not being segmented well. We may also want to look for pixels that are mispredicted, to see if they are spatially correlated (to each other or and (obviously) and to the cell predictions).

- We could use a different segmentation methods (instead of flat threshold)
- The cells could be grown after they are labeled, until they are within 1px of a neighbor.
- We could train on the membrane ground truth *inferred* from the cell segmentations!
- The model could be better! We could continue training it! It wasn't done.

Maybe we want to emphasize the pixels *near* the membrane a little more? By using the weighting scheme from original U-net paper?

Let's run two simulations, both continuing with the model and param_weights from the end of the halfhalf run.
1. Continue with the ground truth data (6x downscaled and cropped?)
2. Continue with the uncropped 6x downscaled data and new associated annotations from cell segmentations directly.
3. Try with only 3x downscaled data and see what kind of accuracy you can get...

-------------------------------------------------------------------------------

# Solved. PROBLEM. Thu Mar  9 18:53:52 2017 -- Image Metadata, API consistency

I've got an api problem.

I've got different labelings all stored in different ways, with different pixel sizes and different meanings for 0,1,2,etc labels. I have to treat these things differently inside my code! Or I have to make (many) copies of all the data to convert everything to a standard format.

At the moment I take care of things inside my code. But the places I have to change are all spread around. I must either:

- move the critical bits that require data-specific changing to a single accessible place.
- Make copies of all the data to a single standardized format.
- Design a tiny description language / metadata format that lives in the folder with the images and is loaded by your program. It will have pixel sizes, intensity-image, pixel-labeling image, segmentation-image, etc. And it will be a data format, so that it can live *with* your images, and doesn't die every time your runtime / program stops. It can also tell us the meaning of the axes in the images! This is better than the metadata that lives inside an individual image, because it can know about the relationship between images inside a folder, and most of the time large microscopy datasets are not stored as a single image file.

In the end I decided to leave the data alone, but automatically apply dataset-specific processing every time the data is loaded. This takes just a few lines in my train.py... It is based on the dataset location, so that isn't allowed to change!

-------------------------------------------------------------------------------

# Fri Mar 10 10:47:18 2017 -- Data Generators, Runtime prediction
[related to: # GPU memory explodes]

I tried running two jobs yesterday on two different versions of the labeled data:

- 3x downscaled copies of the uncropped data. I make 160 x 160 patches with stride 10. This was way too much training data and the first epoch never even began.
- 6x downscaled data with the same 160 x 160 x 10 windowing, which began but required 5000sec for an epoch, so it had only done 10 epochs by the time I came back. AND another funny thing! It had stopped! Because we had the early stopping callback on! We started the learning with the old model and params, and the loss appeared to increase after every round, with the accuracy decreasing all the way down to 89%/86% training/testing.

Potential solutions:
- We need a way of predicting the epoch time and setup time from the windowing. Then we can just keep the windows a reasonable size.
- We use data generators, which allow us to train without building all of X at once and holding it in memory.
- Since running a single 160x160 patch through the net takes about 0.05 secs, we want to load just enough patches in memory that the overhead from repeating this procedure is negligible.

Let's go with the generating-data-as-we-go fix, which is the only way we're going to increase our datasize for working on the fly *anyways*!

We have a few options with the ImageDataGenerator class. When we instantiate one of these objects we tell it which out of a list of potential augmentation methods we want to use, but we can't add our own. Since elastic deformations are *not* available, we won't be able to use them this way. But we can use the 
`model.fit_generator` method which is very similar to our existing `model.fit`.

*How does the generator know what to do with our Y-images?*
*Does the generator know how to tile our images into windows?*

There is `datagen.flow` which takes an unaugmented X,Y and dynamically augemnts them before passing them into the fit_generator method. But there is also the `flow_from_directory` method, which doesn't work for us, because it's expecting the task to be whole-image classification, so it expects one directory full of images per class, and then the labels are inferred from the directory structure. Of course this doesn't work for us.

So we have to first build X,Y and use `datagen.flow` to get dynamic augmentation. Of course, if your dataset is too big already before you even start augmenting it, then you don't avoid any of the memory issues.

Here's what would be ideal:
Dynamically build X,Y in *batches* and perform augmentation dynamically as well... Let's ask Martin/Laurent if they run into this problem.

You can use any generator function you want as long as it returns a full batch each time! That's easy.

We'll build one that samples n patches of size (m,n) from a dataset (a list of heterogeneously-sized images) (optionally with some distribution over the postions i.e. flat across images & flat across locations within an image? or just flat across *all* locations, or weighted to be near membrane? or zero in patches that *don't* include any membrane?)

Aside: The number of generators we want to use maps exactly to our hardware's memory layout! We want a generator for loading from disk → RAM and another generator for loading from RAM → GPU memory!

Martin gave me two functions that I need to test.
Of course they don't work with my current workflow.

I have to save all my X,Y patches to disk to use them... which I could do. But do i want to save them to disk? If I want to sample evenly across all possible 160x160 patches (and then save those to disk) I could. But there may be an advantage to doing it with a perfectly even distribution of points across space, because that's what we want and the model doesn't know anything about space anyways... Of course we could programmatically generate X,Y AND save it to disk every time we run the program. This might be fast! And we can even combine them into a single numpy array with named axes...

# Thoughts: Tue Mar 14 00:04:42 2017 -- Patch Size

I don't know how to choose a patch size.

- It should be as large as possible, to minimize the border effects.
- Or we could only keep the valid region of the convolutions.

How large can they be? What limits the size? Just GPU memory? [[GPU memory explodes]]
Another problem I had was that preprocessing took too long (see previous issue) which depends on the image patch size. Do we really need 16x coverage? 

**I bet the correct patch size is just 3x our model information travel size.** 
**I bet the correct stride is just the information travel size.**

The information travel distance is 29 pixels window size (14 on a side. 2x1 + 2x2 + 2x4 = 14. 14x2 + 1 = 29.) Let's round that to 30 and say 120x120 square patches with stride 30...

We'll apply this to the 6x and 3x downaveraged data.

How much memory will this take? The 160x160 stride 10 patches required 16x coverage, this will only require 3x. Plus if we include the data generator then we can cut it down by another factor of 4x.

# SOLVED: Problem: Tue Mar 14 13:53:54 2017 -- Inability to learn

A couple key realizations... The *Cell_segmentations_paper* folder is not filled with ground truth Annotations! It's filled with the output from the cell segmentation algorithm?! They appear to be the *corrected* output of the cell-segmentation algorithm? Although many bits don't appear to be corrected at all... But in general the output looks quite OK.. Why weren't we able to learn based on these annotations? This is the strangest thing...

Aha! Maybe we couldn't learn because we're not using the generator in the right way! It has to apply the same transformation to both the X AND Y! Does it do that?... hmmmm... yes, it does. We were shuffling the X's and mutating them, but not the ys. the y's were never flipped, etc.

Solution. Related to [[Data Generators]]. Build your own generator! Make sure it transforms both X and Y together.

DONE.

# SOLVED: Sat Mar 18 15:19:16 2017 -- Membranes too thick

Now we mask the data during predictions to remove the worst part of the boundary effects without throwing away all of the partially-valid patch regions.

The predictions look ok, but the membranes are often too wide and we don't seem to do that well when the cells are very small. But it's quite hard to tell actually. When the cells are large the predictions look very confident, although not perfect. There are still some membrane gaps. But they look, in general, quite good.

:TODO:
1. Generate cell and membrane predictions automatically.

This can be improved by decreasing the membrane_weight_multiplier down to 1 again.

# results4/ Summary -- Inability to learn on some types of data...

1. Training on full size data and predicting on 6x downscaled data did not work at all. see seg_full and seg_full_2
2. 

Why does the same net fail to learn on some types of training data in results4/?

Not sure. But the answer to all our problems is to make a database of results. We need to make the parameters and results easily accessible to analysis tools.

# AVOID. Problem: ipython doesn't work on furiosa

It throws some sort of shutil_import error
Oscar doesn't know why. Installed local ipython with pip --user flag.
Now I'm using python3 and I can use the system-provided version.

But ipython STILL doesn't work for a NEW REASON. When I try to import skimage.io it immediately shuts down and complains about X window stuff.

# SOLVED: PyOpenCL doesn't install with pip3 on my local machine

Something to do with architecture problems. I expect the command `pip3 install pyopencl` to install pyopencl. But it fails with the error 

```
ld: warning: ignoring file /usr/local/opt/llvm/lib/libc++.dylib, file was built for x86_64 which is not the architecture being linked (i386): /usr/local/opt/llvm/lib/libc++.dylib
    ld: in '/usr/local/opt/llvm/lib/libunwind.dylib', file was built for x86_64 which is not the architecture being linked (i386): /usr/local/opt/llvm/lib/libunwind.dylib for architecture i386
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    error: command 'g++' failed with exit status 1
```

I thought it might have something to do with my architecture flags. I tried:
ARCHFLAGS="-arch x86_64"
ARCHFLAGS="-arch x86-64"
ARCHFLAGS="-arch=x86_64"
unset ARCHFLAGS

and since none of that changed any of the output I tried it with bash (which seemed to help when I had a similar problem last time I tried to re-install pyopencl on my machine with python2), but using bash didn't help either.

So I'm currently stuck with no good ideas.

SOLVED! We have to install pyopencl by hand, and remove the -arch i386 item from the list of flags. AND we have to install with clang. (which i guess is also what we used to install python3 in the first place...)

# DONE. Indentation Fix for python in vim on the cluster

Fix the indentation and code folding issue on the cluster by changing shiftwidth to 4 `:set sw=4` and setting `:set fdm=indent`.

# SOLVED. Problem: My tensor shape is not what I think it should be. I can't run Theano with channels_first setting.

I'm expecting the "channels" dimension to be after "samples" but before "x", "y" in the theano dimension-ordering configuration.... But it's not! Apparently...

Fixed: They changed the config file args without warning me in Keras2. Now I can only run my script with theano backend, but "image_data_format": "channels_last".

_what's wrong with channels first_ ?

Maybe something is wrong with my loss function?

What is a loss function supposed to take as args? A single patch? or a batch of patches? According to keras, a loss is supposed to take a pair of patches:
(y_true, y_pred). I guess the dimension of each patch is like the dimensionality of Y, but with the samples axis removed... I haven't changed my loss function since the refactor, and I've removed the reshape call... I guess I must have done a reshape both before and after calling the activation softmax? So i guess the problem is not in the loss function. Where is the error coming from?

IGNORE THIS PROBLEM. Just use tensorflow and python3!

PROBLEM SOLVED! I just needed to correct my unet model to allow for a channels_first approach.

# SOLVED! The test score after the re-factor isn't as good as it was pre-refactor. Even though the train score is good!!

Don't worry about recreating old stuff. Your new stuff will be better, after a little work, which you have to put in either way! You should change the loss function so that the pixel weight is a function of the pixels distance from the membrane.

Is this true? Or just a result of switching from hand-labeled membrane to the output of the cellseg pipeline?

NO! Even the output of the classifiers trained on the old hand-labeled membrane look much worse than some of the old, best predictions.

OK, I can load the old model weights after updating the old model to Keras 2. And I can evaluate the loss function with the old classweights. But I don't have the old training/testing data.

# SOLVED: I can't see all my data or make the plots I want.

I want to be able to make plots of test and train score & accuracy for each model as a function of input params, dataset, everything!
I need to save these scores in an easily accessible way. Also, I need to make the differences between models very clear.
I want to use something like tensorboard to see the results of my training. Or do i just want to use the history object? And save it?

- Save the test and train patches and predictions to a directory where you can browse and plot them!
- Save the history object so you can make plots of the score over time!

# SOLVED: PROBLEM: I can't use tensorflow.

When i try to load tensorflow on falcon I get a libcuda not available error.
When I srun myself onto a machine with a gpu on furiosa and do ipython; then import tensorflow; I get a very different error.
`AttributeError: 'module' object has no attribute 'Message'`
from deep inside the core of tensorflow. in resource_handle_pb2.py

Tensorflow is only provided systemwide on furiosa with python3 and you have to be on one of the GPU nodes to avoid the libcuda error.

# PROBLEM: SOLVED: PROBLEM: trying to import train crashes ipython on furiosa

This problem is still a pain in the ass. I can't import skimage.io on ipython because it crashes my X server.

SOLVED: I can avoid this problem (at the cost of very slow startup time) by using -X in my ssh login.

BUT then I run into another problem. I want to connect with -X to a remote node with a GPU via the srun command (queueing system), but I don't know where to put the -X since it's not an ssh call... So I get the same crash as before.
And if I'm NOT on a GPU node, then I cant load tensorflow.

ALSO!!! my jobs fail when I run job_starter.py !!! see training/python3test3/stderr

# ANSWERED. What should loss be? (roughly. numerically. e.g. 3.0 ?)

1. independent of patch size (avg over pixels)
2. independent of number of samples (avg over samples)
3. depends on number of classes (which is usually fixed at 2)
4. Always positive? Its a minus-log-of-a-probability... so it must be positive.

y[samples,x,y,class]

cc = -ytrue * log(ypred)

crossentropy = avg cc across samples,x,y sum across classes.

We should be expecting scores that are more in the 0.01x range (if our accuracy is ~= 95%)!!!

NOTE we can convert between the old loss (non-normalized weights) and the new loss by simply dividing by the normalization factor afterwards (e.g. old weights of {0:1, 1:53} with 1x membrane factor and loss of 0.4 would be 0.4/53=0.0074 in the new loss scheme. With a membrane scale factor of 10x it would be 0.4/(1+53*10)=0.00075329!!!)

# SOLVED. PROBLEM: The old model doesn't even look like it can use the loss function we had!!! 
If the output shape was really (samples, x*y, 2channel) then the loss would see just a 2d (x*y, 2) shaped array, and it would fail!

Crazy stuff. No idea how the old function worked... Also, no idea how the old function trained so well!

!!! Go back and fix b2! That model looked great! What was with those old models? In fact, everything in the results3/ folder looks great! And all of that stuff used basically the same param set...

Hypothesis List
- The last layers of the old model are doing something i don't understand.
  + e.g. the softmax is normalizing over the IMAGE and not the classes! is that possible?
- The loss function is doing something I don't understand when it gets the old input. I really don't see what's happening here, because at the moment i can only get it to throw errors at me...
  + The weights in the loss function are different (does this matter?) It shouldn't because the Adam optimizer normalizes the gradient. so it should only affect the loss, not the magnitude or direction of updates.
- The model params are different in a way that I can't see
- I'm doing some preprocessing of the data that I can't immediately see
  + The old model trains on left and predicts on right halves, so our training data is different. Does this matter?
+ The losses are actually equivalent (but i can't tell because i'm converting the old classweight/normalization incorrectly) and the only difference is in the output of the predictions converted back into images.
- None of the above.
  + Keras1 vs Keras2?
  + Madmax vs Furiosa?
+ 

Experiment 1:
- If I try to change my last layers to copy the old model then I expect it to throw an error because the loss is the wrong shape... Weren't the old Y_predict and Y_train different shapes????!?!?! NO. We also reshaped our Y_train vector to (samples, x*y, 2) for comparison....
Result 1:
The loss function still works when I reshape the x,y dimensions together!

Experiment 2:
Tryin changing to the Theano backend with your reshaped model that handles the 'channel_first' case properly.

Now I'm rerunning a long version of Experiment 1 to see if we can recreate the old high quality output within the same timeframe. 

PARTIALLY SOLVED!!!!

I know how to fix the problem, but I don't understand why the fix works. If I reshape the last layer of my model s.t. I have (samples, x*y, 2) as my Y.shape then training produces a beautiful, clean result. THE LOSS FUNCTION IS THE SAME. How can the same loss function take Y's with different Y.ndim? And why in the world is it *better*?

Hypothesis:
The loss is really an argument of Y's with shape (samples_in_batch, ...) and the ndim=3 loss was secretly doing the wrong thing all the time and ignoring large parts of the data.
Follow-up question:
How do we do sample-based weighting? Because the loss will not have unique id's for each sample across the whole dataset, as you only have access to one batch at a time?....
Result:
IT'S TRUE! The problem since the refactor was that the loss was ignoring a large fraction of each sample, because it was built for ndim=3 input, but I was feeding it ndim=4... Of course when I read the docs on losses on Keras, it looked like they were supposed to take input without as `samples` dimension! Just a single patch! You can see here that the wording is _confusing_ : [https://keras.io/losses/]...

# Solved. Problem. Membranes are too thick

Hypothesis:
The membrane_weight_multiplier was making my membranes extra thick.
I thought that a 10x multiplier was helping me learn, but it was really unnecessary and the results in m56 really show that. All subsequent results confirm this. The membrane width is better when using membrane_weight_multiplier=1.

# Thoughts

- Epochs seem to take around 45±1 seconds for tensorflow.
- A high learning rate can bring faster convergence, but can also lead to sudden, unexpected spikes in loss which take a few epochs to (usually rapidly) recover from.
- The training loss and accuracy aren't necessarily better than the validation! It might be that the validation dataset just happens to be easier to learn, even if it's completely hidden at training time. *can we identify the visible features that make it this way?* *Could it also be that a particular model just happens to perform consistently better on validation than training?* Yes this must also be possible.

**are our models training on exactly the same training data?**

- Yes, i think so. Let's check the train_ind.npy's are the same for both... They are. This means the training patches chosen are the same for both.

# SOLVED: Can't upscale float image to specific image size

You have to use from scipy.misc import imresize, and call it with imresize(..., mode='F')

# Open QUESTION: How do we surpass human performance?

First, we must learn to identify cells as well as an amateur, then as well as an expert. In every case we eliminate the errors that result from carelessness -- simply forgetting to label all parts of the image. And of course, we will surpass human performance in terms of speed. But if all our ground truth labels come from expert|amaterur labelings (with careless forgetting and mislabelings included), how does our accuracy every surpass theirs?

In chess and go the machine's ability to *play* surpassed the human's, but how does this translate to the Computer Vision concepts of accuracy (in identifying objects)? Because the computer doesn't literally see the screen, all of it's decisions are made without worrying about the arbitrary parameters of, e.g: Screen size, Brightness, Contrast, color depth, etc. All of these things affect human accuracy, even when they are focused, desire (are invested in) a high quality outcome, and have expert level knowledge. This means we can train our algorithms on human annotations which have been made by experts under the best possible conditions, taking their time, on the best screens, and adjusting brightness and contrast (other parameters?) until they are fully confident in their decisions. All of this can be built in to the machine.

What about combining the knowledge from different experts?
e.g. in ImageNet you need both expertise in dog breeds and in (maybe?) car types... maybe no one has both! Then the machine is easily the best... (In chess, maybe no grandmaster has both the best mid game and late game...)
Obviously the machine can learn to perform more accurately across the full range of imaging modalities than any single expert. SPIM, confocal, brightfield, fluorescence, phase contrast, etc... But what about within a single dataset? Here all people worthy of the title expert probably perform equally well, (even if their labelings may differ). This removes the possibility of combining the best aspect of different experts. Here improved accuracy must come from one of the inherent advantages to using a machine listed above: speed, consistency = removing dependence on state of mind & attention, removing brightness and contrast as variables (the task of human labeling should always implicitly include this task of adjusting brightness and contrast), and potentially expert-level accuracy.

How does the best machine compare with crowd-sourced decisions? With team-of-experts type decisions? Whenever you need to combine decisions from multiple people you need a voting scheme and/or command hierarchy, which is an independent dimension from the set of people used in the decision.

Image calssifiation as a decoding problem?

# SOLVED: PROBLEM: getting seg, matching, etc scores is too slow

Also, we want the function "equality up to permutation". We can test for this by first building the matching matrix, then seeing if there is a permutation of this matrix that makes it diagonal. And we can do a super-short hack-check, just by testing to see if it is square! If not, then we know the two images are not equivalent up to permutation...

# IGNORE. PROBLEM: How do we augment our labelings?

We need to make sure that the warping doesn't accidentally destroy the nice property that our cell segmentations are given by thresholding the membrane segmentation (with 8-connected cells). *How can we test this?* If we warp the membrane, we expect the number of cells to remain the same!
*They do not.* Thus, we have a problem. With even very small warps, we lose some cells due to the appearance of gaps in the membrane.

*Is this problem important?*

Yeah, we want to be able to scale our ground truth up and down, just like our original images, and our ground truth (in its current form) _relies_ on this property.

_Potential solutions_

- laurent's idea. do the warping on the distance map.
- Convert to svg (whatever continuous representation) before warping

For downscaling, we had a similar problem, but we would only lose cells,
never create them. Cells with diameter smaller than the downscale factor would just disappear. But to solve that we would upscale back to orig size afterwards, and accept that we would just get a certain number of the cells wrong.

solved. I'm doing the warping on the distance maps for labeled images, this helps with the interpolation/warping? How does this compare with just taking all of the non-zero labeled pixels to be foreground (membrane)?

**We can avoid the problem by just not caring about keeping this property for warped images.** Or even by convincing ourselves that the new number of cells is correct! (if we warp membranes of a cell until they touch in the middle, maybe now a two-cell labeling is more appropriate? But if we warp a membrane so that it becomes very thin and breaks then we merge cells... is that ok? After warping we can have large numbers of isolated membrane pixels, which would probably never actually be a labeling we would get from a real user.

todo: test to see what size warping are appropriate for full size images.



# SOLVED. PROBLEM: I want a way of comparing two similar cell labelings visually.

This requires permuting the labels of one images s.t. it aligns closely with another image. We can do this using the matching matrix from label_images, but this problem is very similar to the problem of finding an optimal correspondence between c. elegans nuclei! In our case, however, we only have one ground truth labeling, so we want to see how different the proposed labeling is from it. And also, we don't usually expect to have any warping, so a very flexible matching algorithms doesn't make sense. Usually the proposed solution and the ground truth are built on top of the same underlying greyscale image, so you won't ever find that the proposed solution looks the same as the ground truth, but just translated in x,y... 

Using the 50% overlap criterion is one way of generating a matching
The shared best is another way
Minimizing some difference between centerpoints is another way
There are lots of ways.
A matching is NOT a score!
A score is a number that depends on this matching and on the original images.
A matching is a common intermediate for both proposal/ground-truth scoring, as well as per-cell registration a.la Dave & Dagmar's c. elegan's Atlas Matching.

A good API would make a matching a critical kind of data.

label_imgs.py should provide access to all the stuff we want to do with images where pixels have a label semantic. This includes, coloring, permuting, object matching between two label images, and all the things that are related to matching (many-many and one-one) like finding equivalent labels and computing segmentation error scores.

The graphs we represent as matricies in label_imgs are NOT the typical graph representation! They are rectangular matrices which are only capable of representing bipartite graphs!

_There are multiple ways to represent permutations_ and we need to pick one. - It has to work with numba's @jit decorator. Ideally it would also work with the recoloring of label images that we have in the cell_tracker project.

We can represent them as the bipartite graph matching-matrix (0,1-valued), or with a (n,2) shaped array $arr with just the coordinates of the 1's in that matrix, or with a single array $bb of length $arr[:,0].max() where $bb[i] = $arr[i,1].

We're going to use that third representation for doing the numba permutation, but we want to be able to build it from the first and second representations. In every representation we need to be able to keep knowledge of every existing id in both images. This is encoded in mat.shape. _Do we have this info in the 2nd?_... Making this work in the first and 3rd representation requires that our labels in each image form a continuous sequence [0, 1, ..., n] with no gaps. hmmmmmm. OR we can do it in the first matrix representation... no we can't. we *could* encode it into the pixel_sharing_graph, by saying that row-i label overlaps with zero pixels in the paired image. But then this might be confused with trying to build the graph with images of different sizes.

getting a permutation of type 3 from a matching is easy!

SOLVED. Now we have a few different kinds of matching (that don't try to do any warping) that are useful steps in building the segmentation Error scores, and we can use them to do label permutation as well to aid visual inspection.

NOTE: That this began to touch on the interesting, but mostly unrelated idea of matchings between warped versions of similar images, ala Dagmar's c.elegans matching, but this would be mostly a distraction :)

# PROBLEM: I want to plot my vector fields and warpings on top of my images? How do i do this?

I think you can just plot the vector filed using quiver and streamplot on top of an imshow??? But then we have to make sure we have the axes dimensions right... I want a way of dynamically checking my screen resolution, screen absolute size, and therefore dpi with python!

# SOLVED: PROBLEM: How do I know which of my previous training sessions I should persue?

I need to test my new train/test splitting. I think that the new (correct) splitting will help prevent the overfitting that we've been seeing. I should find a training session that overfit (in few epochs) and retrain with the new datasets.

Search through all datasets that have a history.json. Find the one with the minimum validation loss. This is the best one. Do this for both down3 and orig size images...

The solution to this problem is the same as the solution to the next one. Improve the analysis.py tool so that you always know which model is best and only move the best ones.

# SOLVED: PROBLEM: My training datasets are too big to store on my local machine

How are we going to move our training and testing datasets between local and remote machines? I want to perform analysis on my local machine, because that's where I can use matplotlib. _Does it work on remote if I use ssh -X ?_ NO IT STILL DOESN'T WORK.

Keep a list of the direcories that you've actually analyzed. And keep track of which ones are best.

Solution: I'll use the analysis module to analyze which results are best, and then just move those over to the `training_best` folder.

# cell_tracker PROBLEM. Tra loss might be bad.

When evaluating a loss on a tracking problem we want an error score that doesn't penalize too harshly when correctly identify a division, but shift it backwards or forwards one frame in time wrt the labeling, because the exact timing of a division can be very unclear in real data, and we certainly don't want it to penalize us twice!

# Question: Do I augment before converting to patches, or after?

- Before gives less diversity, but is computationally easier, but might require more code restructuring.

# SOLVED. ERROR: The cluster can't use imread or imresize?

The solution for imread was to specify that plugin='tifffile'. This is not necessary if we upgrade to scikit-image 0.13.0, which we had to do in order to get the correct skimage.morphology.warp function! Also, we got the PIL version of imresize working after installing pillow library.

(Pdb) c
Using TensorFlow backend.
Traceback (most recent call last):
  File "/sw/apps/python/3.5.1/lib/python3.5/pdb.py", line 1661, in main
    pdb._runscript(mainpyfile)
  File "/sw/apps/python/3.5.1/lib/python3.5/pdb.py", line 1542, in _runscript
    self.run(statement)
  File "/sw/apps/python/3.5.1/lib/python3.5/bdb.py", line 431, in run
    exec(cmd, globals, locals)
  File "<string>", line 1, in <module>
  File "/lustre/projects/project-broaddus/carine_smFISH_seg/test.py", line 1, in <module>
    import unet
  File "/lustre/projects/project-broaddus/carine_smFISH_seg/test.py", line 5, in <listcomp>
    lablist = [unet.io.imread(img) for img in glob('data3/labeled_data_cellseg/labels/down3x/*.tif')]
  File "/sw/apps/python/3.5.1/lib/python3.5/site-packages/scikit_image-0.10.1-py3.5-linux-x86_64.egg/skimage/io/_io.py", line 97, in imread
    img = call_plugin('imread', fname, plugin=plugin, **plugin_args)
  File "/sw/apps/python/3.5.1/lib/python3.5/contextlib.py", line 77, in __exit__
    self.gen.throw(type, value, traceback)
  File "/sw/apps/python/3.5.1/lib/python3.5/site-packages/scikit_image-0.10.1-py3.5-linux-x86_64.egg/skimage/io/util.py", line 35, in file_or_url_context
    yield resource_name
  File "/sw/apps/python/3.5.1/lib/python3.5/site-packages/scikit_image-0.10.1-py3.5-linux-x86_64.egg/skimage/io/_io.py", line 97, in imread
    img = call_plugin('imread', fname, plugin=plugin, **plugin_args)
  File "/sw/apps/python/3.5.1/lib/python3.5/site-packages/scikit_image-0.10.1-py3.5-linux-x86_64.egg/skimage/io/manage_plugins.py", line 209, in call_plugin
    return func(*args, **kwargs)
  File "/sw/apps/python/3.5.1/lib/python3.5/site-packages/scikit_image-0.10.1-py3.5-linux-x86_64.egg/skimage/io/_plugins/pil_plugin.py", line 34, in imread
    im = np.fromstring(im.tostring(), dtype)
  File "/home/broaddus/.local/lib/python3.5/site-packages/PIL/Image.py", line 712, in tostring
    raise NotImplementedError("tostring() has been removed. "
NotImplementedError: tostring() has been removed. Please call tobytes() instead

We can fix this problem by making sure that we use the correct version of scikit_image. See tests/warp.py and the way it uses pkg_resources and sys.path. NOTE: that I couldn't find any way having python3 *prefer* my .local site-packages over the main one in /sw/apps without adding it explicitly to the head of the list at runtime. $PYTHONPATH adds it to the END of the list, which is not good enough.

# IGNORE: PROBLEM: divide by zero when combining patchwise predictions.

But I can see many places where I would divide by zero... is this actually a problem? You can divide one numpy array by another, where you divide 0/0 you get a nan. 1/0 gives inf. -1/0 gives -inf. It sometimes gives me a RuntimeWarning, but other times it stops my program!

Dividing by zero leads to nans, which display fine in Fiji, etc. 

The correct solution is to deal with the boundaries to the input s.t. you don't get any nans!

# SOLVED: PROBLEM: prediction still has square artifacts, but only visible on weak/untrained models.

I can still see the squares! This shouldn't happen, or?
Let's write a test...
I want to:
1. take an image apart and put it back together exactly the same way.
2. make sure that the parts of my images that are within the border of completely valid convolutions give exactly the same results from different patches.

These sqaureish / straight-line-ish artifacts are a direct result of the model! Not the process by which the image is broken into patches or sewn back together. Strange! Is this something innate to unets, or is it also present in our training data???

# SOLVED. PROBLEM: I use matplotlib in multiple places. I need a global setting on the cluster that forces me to use the Agg backend!

I don't know how to use a new version of a package I've installed myself without forcibly adding it to the sys.path at the head of the list (this is not platform independent!)

use the matplotlibrc file on the cluster with default backend : Agg

# SOLVED: PROBLEM: I can't save my model architecture because it uses a custom activation function.

This activation function is just a softmax that has it's axis depend on the tensorflow|theano flag! (I could add a conditional permute? but this wouldn't appear in the model!)

I guess I could have my model (and loss) just depend on whether we're using...
Does the conditional permutation already in the code *actually work?*??

Solved.
1. the conditionals works because they are evaulated at model build time.
2. by moving the conditional permute before the softmax I can use the standard softmax function.


# Shitty Workaround: PROBLEM: I can't group my tests to a separate tests folder.

Apparently you aren't supposed to call tests as you would call scripts...
Actually, it looks like preferred behavior is, when making command-line calls within a python project, to use the `python -m module.submodule.subsubmodule` format, and not to just call the file by name e.g. `python tests/warping/test_warp.py`.

If I put scripts into a subdirectory, then I can't import stuff from the parent directory if I call with the filename calling convention, unless you add it to sys.path explicitly.

I avoid this problem by keeping everything in a single directory :)

# IGNORE: PROBLEM: Some of my simple tests require unet, but not keras or any model. Just the generators! The generators have no dependence on keras! Should I separate them?

We've factored out a lot of the unet stuff into patchmaker.py and datasets.py, but we're keeping the generator in there for now. Where should it live? Generators require X,Y and train_params...

# BIG ISSUE FIXED

Fixed the connection bugs in 5&7 layer unets earlier today.
Then found out I had MORE bugs in the number of convolutions in my 7-layer net.
Now I've got an awesome n-layer net. n_pool counts the number of pooling operations.
5 layer was n_pool=2
7 layer was n_pool=3
but we can try 4,5,6, etc!
In the Ronneberger paper they use a network with '23 convolutional layers' which works out to n_pool = 5!

# hacky solution: PROBLEM: $pythonpath env var always has .egg files at the top, so I can't have a default preference for my own installed libraries!

see: 
https://stackoverflow.com/questions/5984523/eggs-in-path-before-pythonpath-environment-variable

It's easy-install's problem, apparently, and I probably can't change that.

The only solution that works for me is to prepend the projects/project-broaddes/.local/ directory to the sys.path list at the start of each script!

# 90% SOLVED: BUG: my patchwork reconstruction produces visible square artifacts

By increasing the step, keeping dx at 480, and keeping itd at 92 I expect to see..... AT LEAST at 20 pix gap between patches.
I can see a 20px gap at the very top of the image. This means my itd is set to 20px! clearly wrong. Does a 20px itd explain the 60px gap between patches? yes... remove 20px twice for itd and add 20px gap from step=500. This means I'm probably setting itd to 20, when I don't mean to...
Found the bug. It was only in "predict.py". We have to set itd before calling predict_single_image, or it will take the default itd, which is 20. Duh.

This did not fully solve the problem! I can still see straight-line artifacts! But they are fewer and further apart.

Another piece of evidence. The bottom gap is 382 pixels. It is possible that the bottom gap is larger, because we only sample patches if they fit entirely within the image. So the largest possible gap would be our patch size-1=479. [and the smallest should be 0+itd=92] Since this cuts off a piece of our image, we'll have to fix it!

And now the patches are 296 wide. This is the correct width according to our formula! Does this mean that our formula for calculating itd is wrong?

I guess that trying a larger itd will fix the problem. How about itd = 140, so patches are 480-2*140 = 200 px wide...

This still produces artifacts!

Now let's try itd=190 and step = 100.

Still artifacts.
Maybe the problem *isn't* my itd, but just the way that I put patches back together... Let's see...

----

New test. Run prediction on patches that are almost entirely overlapping. They should be almost entirely identical... How far do the differences propagate?

----

new hypothesis: I should be using imglist_to_X and normalizing patches before feeding them into predict? Shouldn't this make previously overlapping patches no longer the same? is this what's causing the line artifacts???
- this still doesn't solve the problem of what's causing the differences between Y patches in our test_patch_predictions.py?

new test: Compare a full-image patch prediction with and without normalization. I suspect that removing normalization will get rid of the square artifacts, although it might degrade prediction accuracy. The solution would be to replace all patch-wise normalization with image-wise...

OK, now I've tried with and without normalization. The normalization makes X1!=X2, and of course Y1!=Y2, just like in the no-normalization case.

So even though my Y images fail the test, they look very similar! I can't see any differences. But the difference between the normalized and non-normalized outputs is very large. Turns out there *are* differences, but only a very small number.

I found the differences! (in the non-normalized images) They are near the boundaries of the membrane! All throughout the image! Why is this?

TO TEST: what happens if you send the same patch through multiple times? Does the net produce exactly the same result?

But I *still* see square artifacts in images without any normalization! what is going on here?

This test was with m162, with itd=92 and step = 480-2*92=296. [The width of patches is exactly this 296.]

- can i pull images apart and piece them back together exactly the same as before? yes, i tested this! does the pipeline i use for prediction do exactly this if i use the identity function instead of my normal net?

After fixing/changing patchmaker I know that I can sample patches and put them back together correctly, but I still get square artifacts near the boundaries of my patches. I think it must be due to the normalization. Let's try turning that off (or doing it image-wise) and see what we get...

Normalizing the entire X image as a whole made the problem worse! How can that be... What are the remaining alternatives...

observations... 
1. the problem looks worse outside the main body of the tissue.
2. worse *without* patch-wise normalization...

remaining hypotheses...

1. We're still using patchmaker wrong.
2. The Unet border effects travel further than `itd`.
3. We're changing the patches somehow without knowing it.
4. The unet introduces square artifacts *by iteself*.

*What is the distance between patches we observe?*

*Also, let's test patchmaker on real images, in case they look different.*
*We can even test on a series of convolutions?* No we should test on just setting the border pixels a different color...

After looking closely at the results of pasting squares together and knowing that the calculations for the inter-patch distance are correct I can say the artifacts do not come from the Unet itself and we're not changing the patches... It's either 1 or 2. And it looks like the math for putting patches together is OK. So it really must be #2?

Yes, this has been confirmed by testing the unet module and patchmaker modules together on the cluster. We can use Unet module to create X patches, then put them back together and the result is pixel-for-pixel identical (ignoring left and top boundaries).

So remaining hypotheses are:
1. We don't understand ITD / how large the border should be
2. The model has *learned* to make these square artifacts in cerain situations.

Test: find a region with bad square artifacts. is it on/near the border?
try making the ITD very large. Try running unet prediction centered on that artifact. Actually, just compare the results of two different border width computations. If the border really doesn't matter, (and thus the model has 
square artifacts built in) then they should be pixel-for-pixel identical.

After making test6()...

1. Border size does have an effect on the output, even when border size is greater than the ITD.
2. The differences between the images seem to be stronger where there is signal in the image.

Let's add another hypothesis:
The Unet is not deterministic. The bright lines are coming from the unet and have nothing to do with our borders, but they ALSO have nothing to do with our training data! Maybe somehow a small amount of gpu memory is getting overwritten during prediction/testing?

The next thing to test is to make sure that the output is deterministic, so we get the exact same output if we keep the border size const...

Now, apparently the GPU performs non-deterministic sum reductions... which might be the cause of my straight-line artifacts...

Ran the tests and that's not the issue!

IMPORTANT: Also, there are *clear* horizontal artifacts in the results of a single patch going through the m150 classifier with itd=20... What causes this? This can't be the result of patchmaker, obviously, and with the recent results it also can't be a problem with the GPU. It just looks like I have horizontal artifacts coming from my classifier!

But these are *not* the same horizontal artifacts that I see. In the images I see marks that align with patchborders. They look like misalignments, but it might just be that the function doesn't commute with translations like I thought. I can test this by replacing the unet predict function with one that just destroys patch borders and nothing else.

When using the CPU I see some small differences within the valid regions of two patches that overlap a small amount.

Current Hypotheses:
- Just because there are patch lines with the step spacing doesn't mean there is a problem with patchmaker. There might be a problem with the Unet, or at least our assumptions about what the Unet does.
  + The unet differs when running on CPU and GPU. The GPU may even be partly stochastic.
  + The very simple patch test w the CPU still left some small unexplained differences at the northern boundary.
  + CPU differs from GPU, but still doesn't fix the square artifacts.
  - We know that we can change the boundary regions in any way w/out changing result as long as we don't use Unet module... just internal stuff.
- You are calculating the ITD wrong. You should use the same input sizes as the Unet paper, and only use "valid" convolutions.

And only run things on your local machine.

MAYBE SOLVED:

We were wrong about the ITD, sortof. We calculated the boundary correctly, but we were still missing some interesting piece of info. There is a pyramidal set of square grids imposed on each patch by the 2x2 max-pooling. If these grids are not aligned between neighboring patches, then their predictions will differ. Although in practice these differences are slight, they could in principle be large. In short, the max-pooling breaks the translational symmetry of the convolutions.

TODO: We should be able to run backprop through the net to see which pixels in the input have their weights updated, and how strongly. We really want to be able to analyze a net automatically, with a function to count all the paths in the net from a pixel in the input to one in the output. This is related to visualizing pixel patches that strongly light up a single feature in some middle layer. 

UPDATE:
If we only move our window by 1 pixel, then values all throughout the patch are changed, but if we move by 4 (2^d, where d = n-max-pool layers), then only the boundaries are changed! But the change in the boundaries is greater than we'd expect from our ITD calculations... it was 22 instead of 20. Hmmmm....

SOLVED!

By making the step length between patches an integer multiple of the largest max-pooling pixel size (= 2^d) we get perfectly smooth images!

**But actually the best solution is probably to sample patches from different max-pooling grids and average the results.**

Remaining issues:

- What caused the square artifacts in the individual patches?
  + Hypothesis: failing to normalize X.
+ Where & Why are CPU & GPU different? Is GPU deterministic?
  * Hypothesis: GPU will introduce small random noise in output, uncorrelated with signal.

# PROBLEM: memory easily exhausted when I run tensorflow from iPython (on my mac)

and now CUDNN_STATUS_INTERNAL_ERROR ggguguugggg

# Problem: What kind of improvement do I expect to find with the new data?

I don't know, even roughly, what kind of improvement to expect with the new data. Should the (validation) loss decrease? 

Each of the following problems will cause poor predictions...

How do I know if my model is powerful enough?
- Can it overfit? Use crossvalidation to know when you're overfitting and by how much.
How do I know if I have enough data?
- Does the validation loss saturate as a function of training set size?
How do I know if my data is high quality?
- dunno... If my datapoints overlap in my featurespace, then they are inconsistently labeled, according to those features, so either I need a better featurespace, or I need a better labeling. "label noise" is a term that exists. Usually, we take our labels to be exactly correct, and any overlap in featurespace means we either need a better, more descriptive featurespace, or we've reached the maximum predictive capability of the featurespace + model that we have. If our labels are noisy/meaningless, then even the most powerful/accurate model won't be able to predict well (on training? or vali?) **We can try artificially introducing label noise!** If we introduce different amounts of label noise into the training data then we can see how the prediction quality responds... This should tell us something... If we compare to a perfectly-labeled artifical dataset, then we should be able to figure out how poor our manual-labels are?

# PROBLEM: Saving tiffs in an ImageJ compatible way is Hell.

Is there any way of saving a ZYXC image with C=2? with Uints? Floats?

A float16(36,2,800,800) is interpreted by imagej as (72,800,800) z-stack...
A float16(2,36,800,800) doesn't open, and gives the error "can't open 16-bit floats"...
A uint16 works the same as above.
If I use uint16(36,2,800,800) I can get a z and channel dimensions if I open with BIOformats (huzzah!), but I get the 1st image repeated 72 times if I just drag n' drop... wtf. But not every time! Now BioFormats goes back to reshaping the array as a (large, x, y) patch...

# PROBLEM: After refactoring, I can't learn anything!

I compute reference stakk from available images in a separate step, prior to training. These datasets are computed once, and kept alive for a while, are easy to view & inspect and make sure there are no issues and that the patches are exactly what we want.

Since my local GPU's memory is too small to train anything I resort to doing it remotely, which is fine, because the code is identical, as are the stakks. 

PROBLEM: Predictions are all black or all white.
Hypothesis: All the new data makes my X,Y too big... Or some difference between the old and new data make it difficult to learn them both together? Scale?
Alternative: It's not a data or model problem, but a code problem.
Test: Make a stack exactly the same as the ones we used to run. Use the same model. We should get the same results.

OK, I can't even train on the small simple stack, so I think it must be a bug, and not a problem with the data... All the 

## SOLVED: Subproblem: I get an error when training with old data!

It must be due to the shape or dtype. so print those out....
The dtypes, mins & max are the same, and there's nothing obviously wrong with the shape (it was a shape i'd used before!!!). Still, I'll try cutting down the number of samples to see if that fixes the problem...

Aside: I fixed a bug in getting the size of my training stack...

But still my program crashes with the old training data... So strange!

Check steps_per_epoch... it's correct.

Code "works" with all stakks so far... stakks of width 128 and 512. And from 32 to 3000 samples... The dtypes are the same and the values in the dtypes are the same....

Remaining Hypotheses:
- It has to do with the content of the images themselves? Does x have nans?
- It has to do with the patchwidth: 480... It was never a problem before... but now it is for some reason.

And how is this related to the problem of not being able to learn? No idea...
After several large-ish batches, can we say there's definitely a bug and it's not just failure to train?? I think we can....

Remaining failure to train hypotheses...
- still can't rule out that the new data is keeping us from learning... if anything the 2nd problem adds to that hypothesis...
- could be do the the generalized n_channels

ANOTHER BUG! I hadn't defined the variable `d` (dropout fraction) in my build_unet function before it was used in a function def that captured it... This went silently unnoticed through the keras compile! I have no idea what value `d` had before it was used! And this is an oldddd function that has been used several times to train great networks! Why wasn't this bug caught before!? Either the code changed or the random value that it happened to capture was acceptible... OR this is a red herring and the variable was't capured by value, but by reference and so it didn't need to be defined...
UPDATE :::: NOT A BUG!
The function was not used *before* the variable d was defined, and since closures apparently capture variables by reference (not by value?) (even if undefined at the time) there was no problem when we actually needed to evalue them... So this doesn't explain our inability to learn! Maybe it's only explained by the small datasets and rapid training time?

ORIGINAL PROBLEM GONE. My nets can learn! But now I see the 2nd problem occur with my tiny dataset... it's got something to do with "strided_slice" blah blah... what does this mean????

BUG FOUND! You were, like an idiot, computing the number of classes in to_categorical based on the max value in your Ylabel data, which varies depending on the data!!!! Don't do that.

Remaining hypotheses:
- Inherent randomness of training. Small datasets and short training times.
- there is a problem with my loss function...

TEST: Try training using both loss functions... Or just try evaluating both loss functions on arbitrary data.

Numpy numerical tests return the same floating point values... But when running with the Tensorflow mean,sum,log, etc I get two different results.... The results returned are just the *names* of the functions according to tensorflow, which are generated dynamically. Running the test multiple times returns new names (with increasing numerical string) each time.

# AVOID... PROBLEM: OSError: cannot identify image file 'training/m174/training.tif'

On the falcon server I have this bug when trying to open in ipython with skimage.io, but on my local machine the file opens as int32... Even after changing the save-type to 'uint16' i still have this error....

# PROBLEM: I keep running into key errors when working with dataframes.

This is not a hard problem. It just means fixing the bugs and making the code more robust against changes/ missing data in train_params.json and history.json.




# NOTE: when downsampling from float32 to uint16 we first need to convert values to range [0,2**16-1], otherwise it rolls over.


# TODO:

1. DONE. Fix train/test splitting!
2. DONE. Remove bordermode = 'same' (change to 'valid')
3. DONE. Augment with simple rotation and horizontal reflection
4. DONE. Augment with warping
6. Allow training on arbitrary size dataset. Train on new Data.
5. Compare true cell segmentation scores / old segmentation method / U-net paper w similar dataset?
7. Weight pixels by distance to membrane boundary.
    * Does this still make sense? Given that Ronneberger used this distance penalty mostly to emphasize the background pixels *in between neighboring cells that had a small gap between them*. This is mostly appropriate to single cells-in-a-dish, and not to tissues.
8. Predict only distance maps!
9. Remove one-hot encoding. Predict different cell types. Differentiate background from cytoplasm. Introduce uncertainty via an "i don't know" label. This label must be incorporated into the loss as well!

# Questions | Ideas | Possibly todo

- How can we separate these problems?
  + Model too weak: try overfitting to the max.
  + Not enough data: use datasets of various sizes. see how validation score changes?
  + Data quality poor: randomize different fractions of labels. How robust is validation loss? (Of course, the score will depend not just on the fraction of randomized scores, but on which pixels were randomized. Do we pick pixels w flat distribution across (samples, x, y) ? Or evenly weight by class?)
- If we just training on a single patch, with various augmentations, how good can we do?
- Differentiable cell segmentation losses?
- how much can you warp before ground truth is destroyed?
- complete set of matching, warping and cell seg error measures
- DONE: Laurent's idea: warp the distance fields as opposed to the membranes!
- 

