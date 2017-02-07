from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from glob import glob

import skimage.io as io
import skimage.feature as skf
import skimage.feature.blob as B
import skimage.filters as F

import os

from sklearn.ensemble import RandomForestClassifier

# we want to do an initial classification, then look at object features, then merge cells together as the solution to a graphical model... 
