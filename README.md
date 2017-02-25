# A tool for membrane-labeling and 2D cell segmentation in fluorescence images of mesenchymal tissue sections.

Pic is worth 10^3 words:

![fluorescent membranes, membrane highlighting, cell segmentation](./resources/grey_mem_seg.jpg "fluorescent membranes, membrane highlighting, and cell segmentation")

## About

For information about the sample preparation and imaging involved in creating the data please see [1],[2].

[1]: Stapel, L. Carine, et al. "Automated detection and quantification of single RNAs at cellular resolution in zebrafish embryos." Development 143.3 (2016): 540-546.

[2]: Stapel, L. Carine, et al. (Springer Methods, etc)

## Methods

Membrane pixelwise classification is performed with a 5-layer (2-downsampling, 1-bottom, 2-upsampling) [unet](https://arxiv.org/pdf/1505.04597.pdf) with same model used [here](https://github.com/orobix/retina-unet).

## Installation

You need Python >= 2.7 with the standard scientific python libraries:
- numpy, scipy, scikit-learn and scikit-image
- the deep learning framework Keras
    - which uses as backend: Tensorflow or Theano

## Usage

...
