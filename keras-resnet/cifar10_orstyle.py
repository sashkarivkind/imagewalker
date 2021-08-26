"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from misc import one_hot

import numpy as np
import resnet_a as resnet
import argparse
import os
import time
import resnetpa
import orram_style_nets
lsbjob = os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob

parser = argparse.ArgumentParser()
parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
parser.add_argument('--last_layer_size', default=128, type=int, help='last_layer_size')
parser.add_argument('--dropout1', default=0.2, type=float, help='dropout1')
parser.add_argument('--dropout2', default=0.0, type=float, help='dropout2')
parser.add_argument('--dataset_norm', default=128., type=float, help='dropout2')

parser.add_argument('--layer_norm_res', dest='layer_norm_res', action='store_true')
parser.add_argument('--no-layer_norm_res', dest='layer_norm_res', action='store_false')

parser.add_argument('--layer_norm_2', dest='layer_norm_2', action='store_true')
parser.add_argument('--no-layer_norm_2', dest='layer_norm_2', action='store_false')

parser.add_argument('--skip_conn', dest='skip_conn', action='store_true')
parser.add_argument('--no-skip_conn', dest='skip_conn', action='store_false')

parser.add_argument('--last_maxpool_en', dest='last_maxpool_en', action='store_true')
parser.add_argument('--no-last_maxpool_en', dest='last_maxpool_en', action='store_false')

parser.add_argument('--nl', default='relu', type=str, help='non linearity')

parser.add_argument('--stopping_patience', default=10, type=int, help='stopping patience')
parser.add_argument('--learning_patience', default=5, type=int, help='stopping patience')
parser.add_argument('--manual_suffix', default='', type=str, help='manual suffix')

parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
parser.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false')

parser.add_argument('--rotation_range', default=0.0, type=float, help='dropout1')
parser.add_argument('--width_shift_range', default=0.1, type=float, help='dropout2')
parser.add_argument('--height_shift_range', default=0.1, type=float, help='dropout2')

parser.set_defaults(data_augmentation=True,layer_norm_res=True,layer_norm_2=True,skip_conn=True,last_maxpool_en=True)

config = parser.parse_args()
config = vars(config)
print('config  ',config)
this_run_suffix = lsbjob+'__'+config['manual_suffix']+ str(int(time.time()))

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=config['learning_patience'], min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=config['stopping_patience'])
csv_logger = CSVLogger('resnet_cifar10_{}.csv'.format(this_run_suffix))

batch_size = 32
validate_at_last = 5000
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

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
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_train /= config['dataset_norm']
X_val /= config['dataset_norm']


model =orram_style_nets.parametric_net_befe(dropout1=config['dropout1'],
                                        dropout2=config['dropout2'],
                                        resblocks=config['resblocks'],
                                        layer_norm_res=config['layer_norm_res'],
                                        layer_norm_2=config['layer_norm_2'],
                                        skip_conn=config['skip_conn'],
                                        nl=config['nl'],
                                        last_layer_size=config['last_layer_size'],
                                       last_maxpool_en = config['last_maxpool_en'])

model.compile(loss='sparse_categorical_crossentropy', #todo
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

if not config['data_augmentation']:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
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