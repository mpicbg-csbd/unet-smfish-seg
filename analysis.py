# import skimage.io as io

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from glob import glob
import os
import shutil
import sys
import re

import numpy as np
import pandas as pd

# import importlib.util
import json
from tabulate import tabulate
import segtools as st

dirs_old = glob('training/m?/') + glob('training/m[123456789]?/') + glob('training/m10[01234567]/') # hadn't fixed the val_loss yet
dirs = glob('training/m10[89]/') + glob('training/m1[1234567]?/') # after fixing the val_loss


def explain_training_dir(dr):
    train_params = json.load(open(dr + '/train_params.json'))
    rationale = train_params['rationale']
    history = json.load(open(dr + '/history.json'))
    # print("\n\n")
    # print(dr)
    # print(rationale)
    # print(train_params)
    # print(history['loss'][-1], history['val_loss'][-1])
    # print(history['acc'][-1], history['val_acc'][-1])
    # print("{} epochs in {} seconds".format(len(history['acc']), history['train_time']))
    return rationale, train_params, history

def make_megaplot(dirlist, show=False):
    fig_accuracy = plt.figure()
    axes_accuracy = fig_accuracy.gca()
    fig_loss = plt.figure()
    axes_loss = fig_loss.gca()
    colors = st.pastel_colors_RGB_gap(n_colors=len(dirlist), brightness=1.0)

    # print(colors)
    # x = range(len(colors))
    # plt.figure()
    # plt.scatter(x,x,color=colors)
    # plt.show()

    for i in range(len(dirlist)):
        dr = dirlist[i]
        c = colors[i]
        try:
            history = json.load(open(dr + '/history.json'))
            axes_loss.plot(history['loss'], label='loss_'+dr, color=c)
            axes_loss.plot(history['val_loss'], label='val_loss'+dr, color=c)
            axes_accuracy.plot(history['acc'], label='acc'+dr, color=c)
            axes_accuracy.plot(history['val_acc'], label='val_acc'+dr, color=c)
        except (FileNotFoundError, AttributeError):
            failedlist.append([dr])

    axes_accuracy.legend()
    axes_loss.legend()
    fig_accuracy.savefig('figs/mega_accuracy.pdf')
    fig_loss.savefig('figs/mega_loss.pdf')
    if show:
        fig_accuracy.show()
        fig_loss.show()

def add_plots_to_traindir(dr, show=False):
    history = json.load(open(dr + '/history.json'))
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
    if show:
      plt.show()

def td_summary(dirlist):
    """
    training directory summary
    input: list of training directory names
    output: printed summary table, sorted by loss
    """
    name_acc_loss = []
    failedlist = []
    t_list = []
    h_list = []
    d_list = []

    for i in range(len(dirlist)):
        d = dirlist[i]
        try:
            r,t,h = explain_training_dir(d)
            d_list.append(d) 
            t_list.append(t)
            h_list.append(h)
        except (FileNotFoundError, AttributeError):
            failedlist.append([dirlist[i]])

    df = pd.DataFrame(t_list, index=d_list)
    df2 = pd.DataFrame(h_list, index=d_list)
    df = df.join(df2)
    print(tabulate([["Failed|Ongoing"]] + failedlist))
    return df

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

def info_travel_dist(n_maxpool, conv=3):
    """
    n_maxpool: number of 2x downsampling steps
    conv: the width of the convolution kernel (e.g. "3" for standard 3x3 kernel.)
    returns: the info travel distance == the amount of width that is lost in a patch / 2
    """
    conv2 = 2*(conv-1)
    width = 0
    for i in range(n_maxpool):
        width -= conv2
        width /= 2
    width -= conv2
    for i in range(n_maxpool):
        width *= 2
        width -= conv2
    return int(-width/2)

if __name__ == '__main__':
    df = td_summary(dirs_old + dirs)
    ind = [np.argmin(np.array(val_loss)) for val_loss in df['val_loss']]
    df['ind'] = ind
    df['acc_f'] = [x[i] for x,i in zip(df['acc'], ind)]
    df['loss_f'] = [x[i] for x,i in zip(df['loss'], ind)]
    df['val_acc_f'] = [x[i] for x,i in zip(df['val_acc'], ind)]
    df['val_loss_f'] = [x[i] for x,i in zip(df['val_loss'], ind)]
    df['grey_tif_folder'] = [os.path.normpath(x).split(os.path.sep)[1:] for x in df['grey_tif_folder']]
    df['traindir'] = [int(os.path.normpath(x).split(os.path.sep)[-1][1:]) for x in df.index]
    df.to_pickle('summary.pkl')

