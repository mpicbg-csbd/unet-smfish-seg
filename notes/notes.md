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

It's hard to get exactly the right api. A problem I have at the moment is some functions work at the dataset level (lists of images, full X,Y vectors, etc) and some work at the single image level. X and Y mush image data together, duplicate image data, and can't be pieced back together to make images at all. Maybe we don't want them to be? But probably we do. This requires saving lots of extra info (coordinates for each patch as well as which image those coordinates belong to, as well as keeping the coordinate vector in sync with the X and Y vectors). The way we would do things with Random Forests was to split data up into X/Y_train and X/Y_test. We would calculate our metrics based on the test data, which was fine because each sample in X&Y was an individual pixel, only seen once in the true images. (Although sometimes we would only evaluate a subsample?) But with patch-based classifiers we have duplicated data, so now the accuracy across patches doesn't reflect the accuracy across our images...... No I'm not sure there's a difference, because we still have to break our images up into overlapping patches... In the end we need our images to correspond to classifications, so we must be able to combine the patches in a smart way... TODO: when averaging overlapping patches together this should be done BEFORE we apply the argmax to turn our class-scores into a firm class decision...

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
```

It's important to appreciate that with machine-learned models with enough training data you get systems which are as accurate as *experts* in biological image analysis. Not just random people you hired off of amazon Turk. But people who spend their time looking at (e.g. fluorescence microscopy) images!





