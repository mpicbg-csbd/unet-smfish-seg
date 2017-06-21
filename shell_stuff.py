import glob as g
import os
import shutil
import sys
import re
import numpy as np

import importlib.util
import json
import tabulate


def show_files_gt_1MB(topdir):
    table = []
    for d, ds, fs in os.walk(topdir):
        for f in fs:
            name = d + '/' + f
            size = os.path.getsize(name)/1000/1000 # we're using the metric=ISO standard here
            if size > 1.0:
                table.append([name, '\t', size, "MB"])
    print(tabulate.tabulate(table))


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


def run():
    for d in glob("results3/*p?/"):
        if d[-4] != '_':
            newname = d[:-3] + '_' + d[-3:]
            shutil.move(d, newname)

# change things

def chnames():
    for d in glob("results3/*/"):
        print(d)
        for repl in subdir:
            if repl[0] in d:
                n = 0
                mean = 0
                newname = d.replace(repl[0], repl[1])
                print(newname)
                for img in glob(d + '*.tif'):
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