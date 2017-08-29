from glob import glob
import os
# import shutil
import sys
import re

import json
from tabulate import tabulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
pd.set_option('expand_frame_repr', False)

all_columns = [
    'batch_size',
    'dropout_fraction',
    'epochs',
    'flipLR',
    'grey_tif_folder',
    'initial_model_params',
    'itd',
    'label_tif_folder',
    'learning_rate',
    'membrane_weight_multiplier',
    'model',
    'momentum',
    'n_convolutions_first_layer',
    'patience',
    'rationale',
    'rotate_angle_max',
    'savedir',
    'step',
    'warping_size',
    'x_width',
    'y_width',
    'X_train_shape',
    'X_vali_shape',
    'acc',
    'avg_time_per_batch',
    'avg_time_per_epoch',
    'avg_time_per_sample',
    'loss',
    'steps_per_epoch',
    'train_time',
    'trained_epochs',
    'val_acc',
    'val_loss',
    'warm_up_time',
    ]

show_columns = [
    'batch_size',
    # 'dropout_fraction',
    # 'flipLR',
    # 'step',
    # 'grey_tif_folder',
    'stakk',
    'initial_model_params',
    'learning_rate',
    # 'model',
    # 'np',
    'n_pool',
    'avg_time_per_epoch',
    # 'membrane_weight_multiplier',
    # 'momentum',
    # 'n_convolutions_first_layer',
    'n_conv',
    # 'rotate_angle_max',
    # 'warping_size',
    'X_train_shape',
    'trained_epochs',
    # 'train_time',
    # 'acc',
    # 'val_acc',
    # 'loss',
    # 'val_loss',
    'acc_min',
    'loss_min',
    'val_acc_min',
    'val_loss_min',
    #'traindir',
    ]

plot_columns = [
    # 'acc',
    # 'val_acc',
    'loss',
    'val_loss',
    ]

sort_columns = [
    'val_loss_min',
    'loss_min',
    'acc_min',
    'val_acc_min',
    'traindir',
    ]

dirs_old = glob('training/m?/') + glob('training/m[123456789]?/') + glob('training/m10[01234567]/') # hadn't fixed the val_loss yet
dirs = glob('training/m10[89]/') + glob('training/m1[123456789]?/') + glob('training/m2??') # after fixing the val_loss

def create_df(dirlist):
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
            train_params = json.load(open(d + '/train_params.json'))
            history = json.load(open(d + '/history.json'))
            d_list.append(d)
            t_list.append(train_params)
            h_list.append(history)
        except : #(FileNotFoundError, AttributeError):
            failedlist.append([d])

    df = pd.DataFrame(t_list, index=d_list)
    df2 = pd.DataFrame(h_list, index=d_list)
    df = df.join(df2, lsuffix='_train', rsuffix='_hist')
    print(tabulate([["Failed|Ongoing"]] + failedlist))
    return df

def update_df(df):
    if 'n_convolutions_first_layer' in df.columns:
        df['n_conv'] = df['n_convolutions_first_layer']

    # merge model into n_pool 
    if 'model' in df.columns:
        d = {'unet_5layer':2, 'unet_7layer' : 3}
        ms = df['model']
        ns = df['n_pool']
        df['n_pool'] = [d.get(x,y) for x,y in zip(ms,ns)]

    # merge greytiffolder into stakk 
    def f(greytiffolder, stakk):
        if str(stakk):
            return stakk
        else:
            return os.path.normpath(greytiffolder).split(os.path.sep)[1:]
    if 'grey_tif_folder' in df.columns:
        df['stakk'] = [f(x,y) for x,y in zip(df['grey_tif_folder'], df['stakk'])]

    df['traindir'] = [int(os.path.normpath(x).split(os.path.sep)[-1][1:]) for x in df.index]

    ind = [np.argmin(np.array(val_loss)) for val_loss in df['val_loss']]
    df['ind'] = ind
    df['acc_min']  = [x[i] for x,i in zip(df['acc'], ind)]
    df['loss_min'] = [x[i] for x,i in zip(df['loss'], ind)]
    df['val_acc_min']  = [x[i] for x,i in zip(df['val_acc'], ind)]
    df['val_loss_min'] = [x[i] for x,i in zip(df['val_loss'], ind)]

    #df[df.traindir >= 192][df.traindir <= 212]['val_loss_min'] = "NA"

    def f(imp):
        if imp:
            return imp[:-32]
        return imp
    df['initial_model_params'] = [f(x) for x in df['initial_model_params']]

    df = df.sort_values('traindir')
    return df

def get_n_best(df, n_best=6):
    best = df[df.traindir > 107].sort_values(sort_columns).head(n_best)
    print(best[show_columns])
    return best

def plot_loss(df):
    df[plot_columns].unstack().apply(pd.Series).T.plot()
    # for i in range(len(best)):
    #     row = best.iloc[i]
    #     plt.plot(row['loss'], label=row.name)
    #     plt.plot(row['val_loss'], '--', label=row.name)
    # plt.legend()

def tail(df, n=30):
    print(df[show_columns].tail(n))

def scatterplots(df):
    df.plot.scatter('traindir', 'loss_min')
    df.plot.scatter('traindir', 'val_loss_min')
    df.plot.scatter('traindir', 'train_time')
    df.plot.scatter('train_time', 'val_loss_min')
    df.plot.scatter('loss_min', 'val_loss_min')
    df.plot.scatter('learning_rate', 'avg_time_per_epoch')
    df.plot.scatter('warping_size', 'avg_time_per_epoch')

if __name__ == '__main__':
    df = create_df(dirs)
    df = update_df(df)
    df.to_pickle('summary.pkl')
    show_n_most_recent(df)
