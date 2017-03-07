def check_against_spec(path):
    """The images contained in path must adhere to a strict spec to be used for training.
    This functions checks them against that spec."""

    # must have a `train/` directory with labeled images and featuremaps...
    # feature images should all be of the same type? (Can we make random forests with a
    # mix of numerical and label info?) IF they are not all float images, then we can't
    # use the same set of classifiers/methods. How do we know when an pixel represents a
    # label vs an intensity? Even intensities can be multiple types! floats, ints and
    # uints!

    # tif only has one 'f'

    # The names in the various directories must all align. And the images must be the
    # same size if they share a name.

    # Images *should* have metadata like voxel size and microscopy info. We should be able
    # to include this information in our classifiers. If we want to include this data in
    # our tifs, then it must *also* have its own spec. (essentially a dictionary spec)

    # featurestacks should include the greyscale image. *What about data augmentation?*

    # for classification lables, we should check to make sure that the labels are all
    # the same value and that the values appear in roughly the same distribution!

    # intensity images should all be normalized in the same way. Sometimes float images
    # are forced to take on values in the range [-1, 1] (like in scikit-image!)

    # what about image metadata being stored in filenames? This is obnoxious, but common
    # and useful for quick, but it does tend to blow up path names and seems like a good
    # way to introduce nasty characters into paths... (like spaces!)

    # filenames must be made up of a set of cross-platform standard ascii characters and
    # no spaces, dashes(or?) or slashes. Just upper and lowercase letters, numbers, '_' and '.'.

    # When working with new data, either we edit it s.t. it conforms to our spec, or we
    # change our code s.t. it can read the new data. If you change the data, then your
    # data doesn't look like the original (which is still alive, on someone else's machine
    # but if you change your code, then you have some extra piece of code which has to
    # live in your project made just for reading the new stuff and conforming it to the
    # spec every time you want to load it and run your code. Which way is better? Aha! If
    # you plan on distributing your code, then you can expect that the users will be able
    # to conform their data to your spec, but not that they will be able to add code to
    # read it! This puts the onus on *them* to make your tool work, but shows them exactly
    # how to do it. Putting constraints/expectations in a spec program is *better* than
    # hiding it away in the docs. Users will misread or just not read your docs.
    # Having an (interactive) spec checker will force them to
    # deal with their issues!

    # Number of features in feature stack must be the same across whole directory.

    # x vs y, dim 2 vs dim 1, height vs width, and (left to right vs right left)
    # where is (x,y) = (0,0) ? top right? bottom left? 


    # The label values for pixels that we use with membrane=1/background=0/vertex=2 label images
    # assigns different values to membrane and background than the images with labeled cells
    # where background=1, membrane=0,cell=unique_id...