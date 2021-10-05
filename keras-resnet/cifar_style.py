"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.datasets import cifar10,cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import sys
import os

sys.path.insert(1, os.getcwd()+'/..')

# sys.path.insert(1, '/home/labs/ahissarlab/arivkind/imagewalker')
# sys.path.insert(1, '/home/bnapp/arivkindNet/imagewalker/')

from misc import one_hot

import numpy as np
import resnet_a as resnet
import argparse
import time
import resnetpa
import style_nets
lsbjob = os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob
import cifar10_resnet50_lowResBaseline as cifar10_resnet50
from dataset_utils import bad_res102
parser = argparse.ArgumentParser()
# parser.add_argument('--network_topology', default='default', type=str, help='default or v2')
parser.add_argument('--network_topology', default='default', type=str, help='default or v2')


parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
parser.add_argument('--resblocks16', default=3, type=int, help='resblocks')
parser.add_argument('--resblocks8', default=3, type=int, help='resblocks')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--n_classes', default=100, type=int, help='epochs')
parser.add_argument('--res', default=8, type=int, help='resolution, 0 for default')

parser.add_argument('--last_layer_size', default=128, type=int, help='last_layer_size')
parser.add_argument('--dropout1', default=0.2, type=float, help='dropout1')
parser.add_argument('--dropout2', default=0.0, type=float, help='dropout2')
parser.add_argument('--dataset_norm', default=128., type=float, help='dropout2')

parser.add_argument('--layer_norm_res', dest='layer_norm_res', action='store_true')
parser.add_argument('--no-layer_norm_res', dest='layer_norm_res', action='store_false')
parser.add_argument('--layer_norm_res16', dest='layer_norm_res16', action='store_true')
parser.add_argument('--no-layer_norm_res16', dest='layer_norm_res16', action='store_false')
parser.add_argument('--layer_norm_res8', dest='layer_norm_res8', action='store_true')
parser.add_argument('--no-layer_norm_res8', dest='layer_norm_res8', action='store_false')

parser.add_argument('--layer_norm_2', dest='layer_norm_2', action='store_true')
parser.add_argument('--no-layer_norm_2', dest='layer_norm_2', action='store_false')

parser.add_argument('--skip_conn', dest='skip_conn', action='store_true')
parser.add_argument('--no-skip_conn', dest='skip_conn', action='store_false')
parser.add_argument('--skip_conn16', dest='skip_conn16', action='store_true')
parser.add_argument('--no-skip_conn16', dest='skip_conn16', action='store_false')
parser.add_argument('--skip_conn8', dest='skip_conn8', action='store_true')
parser.add_argument('--no-skip_conn8', dest='skip_conn8', action='store_false')


parser.add_argument('--dense_interface', dest='dense_interface', action='store_true')
parser.add_argument('--no-dense_interface', dest='dense_interface', action='store_false')

parser.add_argument('--last_maxpool_en', dest='last_maxpool_en', action='store_true')
parser.add_argument('--no-last_maxpool_en', dest='last_maxpool_en', action='store_false')

parser.add_argument('--nl', default='relu', type=str, help='non linearity')

parser.add_argument('--stopping_patience', default=10, type=int, help='stopping patience')
parser.add_argument('--learning_patience', default=5, type=int, help='stopping patience')
parser.add_argument('--manual_suffix', default='', type=str, help='manual suffix')

parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
parser.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false')

parser.add_argument('--compile_on_spot', dest='compile_on_spot', action='store_true')
parser.add_argument('--no-compile_on_spot', dest='compile_on_spot', action='store_false')

parser.add_argument('--rotation_range', default=0.0, type=float, help='dropout1')
parser.add_argument('--width_shift_range', default=0.1, type=float, help='dropout2')
parser.add_argument('--height_shift_range', default=0.1, type=float, help='dropout2')

parser.set_defaults(data_augmentation=True,
                    layer_norm_res=True,layer_norm_res16=True,layer_norm_res8=True,
                    layer_norm_2=True,
                    skip_conn=True,skip_conn8=True,skip_conn16=True,
                    dense_interface=False,
                    last_maxpool_en=True,
                    compile_on_spot=True)

config = parser.parse_args()
config = vars(config)
print('config  ',config)
this_run_suffix = lsbjob+'__'+config['manual_suffix']+ str(int(time.time()))

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=config['learning_patience'], min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=config['stopping_patience'])
csv_logger = CSVLogger('resnet_cifar10_{}.csv'.format(config['n_classes'],this_run_suffix))

batch_size = 32
validate_at_last = 5000
nb_classes = config['n_classes']
nb_epoch = config['epochs']
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
if config['n_classes']==10:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
elif config['n_classes']==100:
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
else:
    error

if config['res']!=-1:
    X_train = np.array([bad_res102(xx, (config['res'], config['res'])) for xx in X_train])
    X_test = np.array([bad_res102(xx, (config['res'], config['res'])) for xx in X_test])

# Convert class vectors to binary class matrices.
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

Y_train = y_train[:-validate_at_last]
Y_val = y_train[-validate_at_last:]

# Y_train = one_hot(y_train, nb_classes)
# Y_test = one_hot(y_test, nb_classes)

X_val = X_train[-validate_at_last:].astype('float32')
X_train = X_train[:-validate_at_last].astype('float32')

# subtract mean and normalize
if  config['network_topology'] == 'resnet50_on_imagenet':
    X_train = cifar10_resnet50.preprocess_image_input(X_train)
    X_val = cifar10_resnet50.preprocess_image_input(X_val)
else:
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_train /= config['dataset_norm']
    X_val /= config['dataset_norm']


if config['network_topology'] == 'resnet50_on_imagenet':
    model = cifar10_resnet50.define_compile_split_model(res=(config['res'] if config['res']!=-1 else 32),
                                                        metrics=['sparse_categorical_accuracy'],
                                                        n_classes=config['n_classes'])
    # model = orram_style_nets.parametric_net_befe_v2(dropout1=config['dropout1'],
    #                                                 dropout2=config['dropout2'],
    #                                                 resblocks16=config['resblocks16'],
    #                                                 resblocks8=config['resblocks8'],
    #                                                 layer_norm_res16=config['layer_norm_res16'],
    #                                                 layer_norm_res8=config['layer_norm_res8'],
    #                                                 layer_norm_2=config['layer_norm_2'],
    #                                                 skip_conn16=config['skip_conn16'],
    #                                                 skip_conn8=config['skip_conn8'],
    #                                                 dense_interface=config['dense_interface'],
    #                                                 last_maxpool_en=config['last_maxpool_en'],
    #                                                 nl=config['nl'],
    #                                                 last_layer_size=config['last_layer_size'])
elif config['network_topology'] == 'v2':
    model =style_nets.parametric_net_befe_v2(dropout1=config['dropout1'],
                                                   dropout2=config['dropout2'],
                                                   resblocks16=config['resblocks16'],
                                                   resblocks8=config['resblocks8'],
                                                   layer_norm_res16=config['layer_norm_res16'],
                                                   layer_norm_res8=config['layer_norm_res8'],
                                                   layer_norm_2=config['layer_norm_2'],
                                                   skip_conn16=config['skip_conn16'],
                                                   skip_conn8=config['skip_conn8'],
                                                   dense_interface=config['dense_interface'],
                                                   last_maxpool_en=config['last_maxpool_en'],
                                                   nl=config['nl'],
                                                   last_layer_size=config['last_layer_size'])
elif config['network_topology'] == 'default':
    model =style_nets.parametric_net_befe(dropout1=config['dropout1'],
                                            dropout2=config['dropout2'],
                                            resblocks=config['resblocks'],
                                            layer_norm_res=config['layer_norm_res'],
                                            layer_norm_2=config['layer_norm_2'],
                                            skip_conn=config['skip_conn'],
                                            dense_interface=config['dense_interface'],
                                            nl=config['nl'],
                                            last_layer_size=config['last_layer_size'],
                                            last_maxpool_en = config['last_maxpool_en'])
else:
    error

if config['compile_on_spot']:
    model.compile(loss='sparse_categorical_crossentropy', #todo
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy'])

if not config['data_augmentation']:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(X_val, Y_val),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=config['rotation_range'],  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=config['width_shift_range'],  # randomly shift images horizontally (fraction of total width)
        height_shift_range=config['height_shift_range'],  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_val, Y_val),
                        epochs=nb_epoch, verbose=2, # max_q_size=100,   #todo
                        callbacks=[lr_reducer, early_stopper, csv_logger])

model.save('model_{}.hdf'.format(this_run_suffix))