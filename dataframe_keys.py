import pandas as pd

biglist = [
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

show_these = [
    'batch_size',
    # 'dropout_fraction',
    # 'flipLR',
    'step',
    'grey_tif_folder',
    'initial_model_params',
    'learning_rate',
    'model',
    'np',
    'n_pool',
    'avg_time_per_epoch',
    # 'membrane_weight_multiplier',
    # 'momentum',
    # 'n_convolutions_first_layer',
    # 'rotate_angle_max',
    'warping_size',
    'X_train_shape',
    'trained_epochs',
    'train_time',
    # 'acc',
    # 'val_acc',
    # 'loss',
    # 'val_loss',
    'acc_f',
    'loss_f',
    'val_acc_f',
    'val_loss_f',
    # 'traindir',
    ]

sort_by_these = [
    'val_loss_f',
    'loss_f',
    'traindir',
    'acc_f',
    'val_acc_f',
    ]


import matplotlib.pyplot as plt
plt.ion()

df = pd.read_pickle('summary.pkl')

def add_npool(df):
    d = {'unet_5layer':2, 'unet_7layer' : 3}
    m = df['model']
    n = df['n_pool']
    df['np'] = [d.get(mi, ni) for mi,ni in zip(m,n)]

def display_best(n_best=6):
    plt.ion()
    df = pd.read_pickle('summary.pkl')
    df = add_npool(df)
    pd.set_option('expand_frame_repr', False)
    best = df[df.traindir > 107].sort_values(sort_by_these).iloc[:n_best]
    print(best[show_these])
    for i in range(len(best)):
        row = best.iloc[i]
        plt.plot(row['loss'], label=row.name)
        plt.plot(row['val_loss'], '--', label=row.name)
    plt.legend()

def scatterplots():
    df.plot.scatter('traindir', 'loss_f')
    df.plot.scatter('traindir', 'val_loss_f')
    df.plot.scatter('train_time', 'val_loss_f')
    df.plot.scatter('loss_f', 'val_loss_f')
    df.plot.scatter('learning_rate', 'avg_time_per_epoch')
    df.plot.scatter('warping_size', 'avg_time_per_epoch')

def summary_text():
    # os.path.normpath(a).split(os.path.sep)
    # Remember that directories starting with m108 have the correct val_loss
    df = pd.read_pickle('summary.pkl')
    add_npool(df)
    pd.set_option('expand_frame_repr', False)
    best = df.sort_values(sort_by_these)[show_these][df.traindir > 157]
    print(best)
    top = df.sort_values(sort_by_these)[df.traindir > 157].iloc[0]
    return top






