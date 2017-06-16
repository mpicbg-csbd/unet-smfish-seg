

import skimage.io as io
import matplotlib.pyplot as plt

import glob as g
import os
import shutil
import sys
import re
import numpy as np

import importlib.util
import json


def explain_training_dir(dr):
    spec = importlib.util.spec_from_file_location("train", dr + '/train.py')
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    rationale = foo.rationale
    train_params = foo.train_params
    history = json.load(open(dr + '/history.json'))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(dr + '/loss.pdf')
    plt.figure()
    plt.plot(history['acc'], label='acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.savefig(dr + '/accuracy.pdf')
    return rationale, train_params, history


def walkfiles():
    "Walk through data3/ directories and explain them."
    for d in os.walk("./", topdown=False):
        if 'labels' not in d[0]:
            print(d[0])
            tifs = [s for s in d[2] if s[-4:]=='.tif']
            xsizes = set()
            print("Tif count: ", len(tifs))
            if len(tifs) > 0:
                print(tifs)
                if "SPC0_TM0057_CM0_CM1_CHN00_CHN01.fusedStack.tif" in set(tifs):
                    print("FOUND IT: ", d[0])
                    break
                # n,meanx,meany = 0,0,0
                # for file in tifs:
                #   xdim = io.imread(d[0] + '/' + file).shape[1]
                #   meanx += xdim
                #   ydim = io.imread(d[0] + '/' + file).shape[0]
                #   meany += ydim
                #   n += 1
                #   xsizes.add(xdim)
                # print "xmean, ymean: ", int(float(meanx) / n), ',', int(float(meany) / n)
                # print "sizes: ", xsizes
            print()

def combine(dir1, dir2):
    imgs1 = g.glob(dir1 + '/*.tif')
    imgs2 = g.glob(dir2 + '/*.tif')
    try:
        os.makedirs('combined/')
    except:
        pass
    for i1,i2 in zip(imgs1, imgs2):
        img1 = io.imread(i1)
        img2 = io.imread(i2)
        res = np.stack((img1, img2), axis=0)
        print(i1, '\n', i2, end=' ')
        print("res has shape: ", res.shape, '\n')
        print('dtype: ', img1.dtype, img2.dtype)
        io.imsave('combined/' + os.path.basename(i1), res)

def explain_results(dir):
    "Look at the success rate of jobs as a function of X.shape"
    for d, sub_ds, files in os.walk(dir, topdown=True):
        print(d)
        try:
           for line in open(d + '/rational.txt','r'):
               print(line, end=' ')
        except:
            print("no rational.txt")
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

def run():
    for d in g.glob("results3/*p?/"):
        if d[-4] != '_':
            newname = d[:-3] + '_' + d[-3:]
            shutil.move(d, newname)

# change things

def chnames():
    for d in g.glob("results3/*/"):
        print(d)
        for repl in subdir:
            if repl[0] in d:
                n = 0
                mean = 0
                newname = d.replace(repl[0], repl[1])
                print(newname)
                for img in g.glob(d + '*.tif'):
                    imgshape = io.imread(img).shape
                    # print img, imgshape
                    mean += imgshape[1]
                    n += 1
                print("mean is ", float(mean)/n)
                shutil.move(d, newname)
        print()

def movefiles():
    for d in newdir:
        nd1 = os.path.dirname(d)
        nd = os.path.dirname(nd1)
        nd = nd + '/' + newdir[d] + '/'
        print(nd)
        print(nd1)
        shutil.move(nd1, nd)

if __name__ == '__main__':
    explain_results(sys.argv[1])
