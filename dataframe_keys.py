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
    # 'model',
    # 'membrane_weight_multiplier',
    # 'momentum',
    # 'n_convolutions_first_layer',
    # 'rotate_angle_max',
    # 'warping_size',
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
    'loss_f',
    'val_loss_f',
    'traindir',
    'acc_f',
    'val_acc_f',
    ]


def display_best():
    import matplotlib.pyplot as plt
    plt.ion()
    df = pd.read_pickle('summary.pkl')
    pd.set_option('expand_frame_repr', False)
    best = df[df.traindir > 107].sort_values(sort_by_these).iloc[:6]
    print(best[show_these])
    for i in range(len(best)):
        row = best.iloc[i]
        print(type(row))
        plt.plot(row['loss'], label=row.name)
        plt.plot(row['val_loss'], '--', label=row.name)
    plt.legend()

if False:
    df.plot.scatter('traindir', 'loss_f')
    df.plot.scatter('traindir', 'val_loss_f')
    df.plot.scatter('train_time', 'val_loss_f')

if False:
    # os.path.normpath(a).split(os.path.sep)
    # Remember that directories starting with m108 have the correct val_loss
    df = pd.read_pickle('summary.pkl')
    pd.set_option('expand_frame_repr', False)
    print(df.sort_values(sort_by_these)[show_these][df.traindir > 107])
    df[df.traindir > 107].plot.scatter('traindir', 'val_loss_f')
    # plt.figure()
    df[df.traindir > 107].plot.scatter('loss_f', 'val_loss_f')