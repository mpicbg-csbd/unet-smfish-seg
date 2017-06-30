# import skimage.io as io
import matplotlib.pyplot as plt

from glob import glob
import os
import shutil
import sys
import re
import numpy as np

import importlib.util
import json
import tabulate
import label_imgs

def explain_training_dir(dr, plots=True, megaplots_axes=None):
    spec = importlib.util.spec_from_file_location("train", dr + '/train.py')
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    rationale = foo.rationale
    train_params = foo.train_params
    history = json.load(open(dr + '/history.json'))
    axes_accuracy, axes_loss, color = megaplots_axes
    if plots:
        plt.figure()
        plt.title(dr)
        plt.plot(history['loss'], label='loss_')
        plt.plot(history['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig(dr + '/loss.pdf')
        plt.figure()
        plt.title(dr)
        plt.plot(history['acc'], label='acc')
        plt.plot(history['val_acc'], label='val_acc')
        plt.legend()
        plt.savefig(dr + '/accuracy.pdf')
        plt.show()
    if megaplots_axes:
        axes_loss.plot(history['loss'], label='loss_'+dr, color=color)
        axes_loss.plot(history['val_loss'], label='val_loss'+dr, color=color)
        axes_accuracy.plot(history['acc'], label='acc'+dr, color=color)
        axes_accuracy.plot(history['val_acc'], label='val_acc'+dr, color=color)
    print("\n\n")
    print(dr)
    print(rationale)
    print(train_params)
    print(history['loss'][-1], history['val_loss'][-1])
    print(history['acc'][-1], history['val_acc'][-1])
    print("{} epochs in {} seconds".format(len(history['acc']), history['train_time']))
    return rationale, train_params, history

def explain_training_directories(dir_or_list_of_dirs, plots=True):
    """
    input: parent directory name or list of training directory names.
    output: plots comparing accuracy and loss over time. prints params, timings & rationale.
    """
    if type(dir_or_list_of_dirs)==str:
        dir_or_list_of_dirs = sorted(glob(basedir + '*/'))

    fig_accuracy = plt.figure()
    axes_accuracy = fig_accuracy.gca()
    fig_loss = plt.figure()
    axes_loss = fig_loss.gca()

    print("LENGTH: ", len(dir_or_list_of_dirs))
    colors = label_imgs.pastel_colors_RGB_gap(n_colors=len(dir_or_list_of_dirs), brightness=1.0)
    print(colors)
    x = range(len(colors))
    plt.figure()
    plt.scatter(x,x,color=colors)
    plt.show()

    for i in range(len(dir_or_list_of_dirs)):
        d = dir_or_list_of_dirs[i]
        try:
            r,t,h = explain_training_dir(d, plots=False, megaplots_axes=(axes_accuracy, axes_loss, colors[i]))
        except (FileNotFoundError, AttributeError):
            pass

    axes_accuracy.legend()
    fig_accuracy.show()
    axes_loss.legend()
    fig_loss.show()

def explain_results(dir):
    "Get scores, runtime, etc for old-style training without a history.json or train_params.json"
    for d in glob(dir + '*/'):
        print(d)
        try:
           for line in open(d + '/rationale.txt','r'):
               print(line, end=' ')
        except:
            print("no rationale.txt")
        try:
            epochNum = ""
            for line in open(d + '/stdout','r'):
                if re.search('X.shape|CPU time|accuracy|score', line):
                    print(line, end=' ')
                if re.search(r'Epoch', line):
                    epochNum = line
            print(epochNum, end=' ')
            for line in open(d + '/main.py','r'):
                if re.search('^\s+greys =|greys_imgs =', line):
                    print(line, end=' ')
        except IOError:
            print("IOerror")
        print()

if __name__ == '__main__':
    explain_results(sys.argv[1])


def info_travel_dist(layers, conv=3):
    """
    layers: number of down and up layers (e.g. two down followed by two up => layers=2)
    conv: the width of the convolution kernel (e.g. "3" for standard 3x3 kernel.)
    returns: the info travel distance == the amount of width that is lost in a patch / 2
    """
    conv2 = 2*(conv-1)
    width = 0
    for i in range(layers):
        width -= conv2
        width /= 2
    width -= conv2
    for i in range(layers):
        width *= 2
        width -= conv2
    return -width/2





