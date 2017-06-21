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


def explain_training_dir(dr, plots=True):
    spec = importlib.util.spec_from_file_location("train", dr + '/train.py')
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    rationale = foo.rationale
    train_params = foo.train_params
    history = json.load(open(dr + '/history.json'))
    plt.figure()
    plt.title(dr)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(dr + '/loss.pdf')
    plt.figure()
    plt.title(dr)
    plt.plot(history['acc'], label='acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.savefig(dr + '/accuracy.pdf')
    if plots:
        plt.show()
    print("\n\n")
    print(dr)
    print(rationale)
    print(train_params)
    print(history['loss'][-1], history['val_loss'][-1])
    print(history['acc'][-1], history['val_acc'][-1])
    return rationale, train_params, history

def explain_training_directories(dir_or_list_of_dirs):
    if type(dir_or_list_of_dirs)==str:
        dir_or_list_of_dirs = sorted(glob(basedir + '*/'))
    for d in dir_or_list_of_dirs:
        try:
            r,t,h = explain_training_dir(d, plots=True)
        except (FileNotFoundError, AttributeError):
            pass

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
