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

I have a better now of why Dave's CRF wont work on linux. The `learn` binary requires a very special commit from Dave's code. We don't know which one, because we never got the source, only the results of the build. We could try building on every commit, and running tests until we find one that passes the tests... But it will very likely be the wrong commit...

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

## DAWN OF THE DEEP U-NETS

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


















PROBLEMS. IN CHRONOLOGICAL ORDER.

# GPU memory explodes

:TODO:
You can fix this by figuring out the function (dataset,model) → (Max GPU memory usage, Max RAM Usage, Time taken). This shouldn't even be that hard to do!

# Thu Mar  9 13:28:55 2017 -- Early stopping, weak models & proper scaling

I submitted a job to the cluster last night: `results/bigim3/`. It was supposed to run for 300 epochs, but only ran for 18. There were no signs of error in the stderr log, or in stdout.

The callbacks I was running include...
checkpoint and earlystopper.

I bet it was earlystopper on the validation loss, which appeared not to be advancing very much any more.

I changed the verbosity to 1, so it will tell me when it causes an earlystop, but I aslo bet there was a problem that the model was just not powerful enough.

We can either try to fix this problem by adding more layers, or by decreasing the size of the input images. Our normal size is 6x smaller. This leads to good predictions, but we have trouble upscaling the images again.

If we're going to get good cell segmentations, we don't want to be upscaling images by a factor of 6!!! This will completely destroy many of the cells in the ground truth, turning them into things only a few pixels on a side...

We want to know how: *how much can we downscale the image and still get 100% correct cell segmentations after upscaling?* We can answer this question (and the maximum accuracy of our membrane segmentations) because we have access to cell and membrane segmentation ground truth.

-------------------------------------------------------------------------------

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

# Thu Mar  9 18:53:52 2017 -- Image Metadata, API consistency

I've got an api problem.

I've got different labelings all stored in different ways, with different pixel sizes and different meanings for 0,1,2,etc labels. I have to treat these things differently inside my code! Or I have to make (many) copies of all the data to convert everything to a standard format.

At the moment I take care of things inside my code. But the places I have to change are all spread around. I must either:

- move the critical bits that require data-specific changing to a single accessible place.
- Make copies of all the data to a single standardized format.
- Design a tiny description language / metadata format that lives in the folder with the images and is loaded by your program. It will have pixel sizes, intensity-image, pixel-labeling image, segmentation-image, etc. And it will be a data format, so that it can live *with* your images, and doesn't die every time your runtime / program stops. It can also tell us the meaning of the axes in the images! This is better than the metadata that lives inside an individual image, because it can know about the relationship between images inside a folder, and most of the time large microscopy datasets are not stored as a single image file.

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

# Tue Mar 14 00:04:42 2017 -- Patch Size

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

# Tue Mar 14 13:53:54 2017 -- Inability to learn -- SOLVED

A couple key realizations... The *Cell_segmentations_paper* folder is not filled with ground truth Annotations! It's filled with the output from the cell segmentation algorithm?! They appear to be the *corrected* output of the cell-segmentation algorithm? Although many bits don't appear to be corrected at all... But in general the output looks quite OK.. Why weren't we able to learn based on these annotations? This is the strangest thing...

Aha! Maybe we couldn't learn because we're not using the generator in the right way! It has to apply the same transformation to both the X AND Y! Does it do that?... hmmmm... yes, it does. We were shuffling the X's and mutating them, but not the ys. the y's were never flipped, etc.

Solution. Related to [[Data Generators]]. Build your own generator! Make sure it transforms both X and Y together.

DONE.

# Sat Mar 18 15:19:16 2017 -- Membranes too thick

Now we mask the data during predictions to remove the worst part of the boundary effects without throwing away all of the partially-valid patch regions.

The predictions look ok, but the membranes are often too wide and we don't seem to do that well when the cells are very small. But it's quite hard to tell actually. When the cells are large the predictions look very confident, although not perfect. There are still some membrane gaps. But they look, in general, quite good.

:TODO:
1. Generate cell and membrane predictions automatically.
2. 

# results4/ Summary -- Inability to learn on some types of data...

1. Training on full size data and predicting on 6x downscaled data did not work at all. see seg_full and seg_full_2
2. 

Why does the same net fail to learn on some types of training data in results4/?

Not sure. But the answer to all our problems is to make a database of results. We need to make the parameters and results easily accessible to analysis tools.

# TEMPFIX: ipython doesn't work on furiosa

It throws some sort of 
Oscar doesn't know why. Installed local ipython with pip --user flag.

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

# Problem: My tensor shape is not what I think it should be. I can't run Theano with channels_first setting.

I'm expecting the "channels" dimension to be after "samples" but before "x", "y" in the theano dimension-ordering configuration.... But it's not! Apparently...

Fixed: They changed the config file args without warning me in Keras2. Now I can only run my script with theano backend, but "image_data_format": "channels_last".

_what's wrong with channels first_ ?

Maybe something is wrong with my loss function?

What is a loss function supposed to take as args? A single patch? or a batch of patches? According to keras, a loss is supposed to take a pair of patches:
(y_true, y_pred). I guess the dimension of each patch is like the dimensionality of Y, but with the samples axis removed... I haven't changed my loss function since the refactor, and I've removed the reshape call... I guess I must have done a reshape both before and after calling the activation softmax? So i guess the problem is not in the loss function. Where is the error coming from?

# The test score after the re-factor isn't as good as it was pre-refactor. Even though the train score is good.

Don't worry about recreating old stuff. Your new stuff will be better, after a little work, which you have to put in either way! You should change the loss function so that the pixel weight is a function of the pixels distance from the membrane.

# I can't see all my data or make the plots I want.

I want to be able to make plots of test and train score & accuracy for each model as a function of input params, dataset, everything!
I need to save these scores in an easily accessible way. Also, I need to make the differences between models very clear.
I want to use something like tensorboard to see the results of my training. Or do i just want to use the history object? And save it?

- Save the test and train patches and predictions to a directory where you can browse and plot them!
- Save the history object so you can make plots of the score over time!
- 

# I can't use tensorflow.
When i try to load tensorflow on falcon I get a libcuda not available error.
When I srun myself onto a machine with a gpu on furiosa and do ipython; then import tensorflow; I get a very different error.
`AttributeError: 'module' object has no attribute 'Message'`
from deep inside the core of tensorflow. in resource_handle_pb2.py




















