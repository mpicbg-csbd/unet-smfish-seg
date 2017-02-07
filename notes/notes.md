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

RESULT! from crf_vs_xgboost.py
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
