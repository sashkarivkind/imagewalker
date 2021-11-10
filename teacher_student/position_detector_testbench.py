#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
added the explore relations part after 735561
"""

import os 
import sys
import gc
sys.path.insert(1, os.getcwd()+'/..')
sys.path.insert(1, os.getcwd()+'/../keras-resnet/')
# sys.path.insert(1, '/home/labs/ahissarlab/arivkind/imagewalker')

# sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
# sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, cifar100
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import time
import pickle
import argparse
from feature_learning_utils import  student3,student4,student5,  write_to_file, traject_learning_dataset_update,  net_weights_reinitializer, load_student, pos_det101
from keras_utils import create_cifar_dataset, split_dataset_xy
from dataset_utils import Syclopic_dataset_generator, test_num_of_trajectories
import cifar10_resnet50_lowResBaseline as cifar10_resnet50
print(os.getcwd() + '/')
#%%

parser = argparse.ArgumentParser()

#general parameters
parser.add_argument('--run_name_prefix', default='noname', type=str, help='path to pretrained teacher net')
parser.add_argument('--run_index', default=10, type=int, help='run_index')
parser.add_argument('--verbose', default=2, type=int, help='run_index')

parser.add_argument('--n_classes', default=10, type=int, help='classes')

parser.add_argument('--testmode', dest='testmode', action='store_true')
parser.add_argument('--no-testmode', dest='testmode', action='store_false')

### student parameters
parser.add_argument('--epochs', default=5, type=int, help='num training epochs')
parser.add_argument('--int_epochs', default=1, type=int, help='num internal training epochs')
parser.add_argument('--decoder_epochs', default=40, type=int, help='num internal training epochs')
parser.add_argument('--num_feature', default=64, type=int, help='legacy to be discarded')
parser.add_argument('--rnn_layer1', default=32, type=int, help='legacy to be discarded')
parser.add_argument('--rnn_layer2', default=64, type=int, help='legacy to be discarded')
parser.add_argument('--time_pool', default=0, help='time dimention pooling to use - max_pool, average_pool, 0')

parser.add_argument('--student_block_size', default=1, type=int, help='number of repetition of each convlstm block')
parser.add_argument('--upsample', default=0, type=int, help='spatial upsampling of input 0 for no')


parser.add_argument('--conv_rnn_type', default='lstm', type=str, help='conv_rnn_type')
parser.add_argument('--student_nl', default='relu', type=str, help='non linearity')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout1')
parser.add_argument('--rnn_dropout', default=0.0, type=float, help='dropout1')
parser.add_argument('--pretrained_student_path', default=None, type=str, help='pretrained student, works only with student3')

parser.add_argument('--decoder_optimizer', default='Adam', type=str, help='Adam or SGD')

parser.add_argument('--skip_student_training', dest='skip_student_training', action='store_true')
parser.add_argument('--no-skip_student_training', dest='skip_student_training', action='store_false')

parser.add_argument('--fine_tune_student', dest='fine_tune_student', action='store_true')
parser.add_argument('--no-fine_tune_student', dest='fine_tune_student', action='store_false')

parser.add_argument('--layer_norm_student', dest='layer_norm_student', action='store_true')
parser.add_argument('--no-layer_norm_student', dest='layer_norm_student', action='store_false')

parser.add_argument('--batch_norm_student', dest='batch_norm_student', action='store_true')
parser.add_argument('--no-batch_norm_student', dest='batch_norm_student', action='store_false')

parser.add_argument('--val_set_mult', default=5, type=int, help='repetitions of validation dataset to reduce trajectory noise')


### syclop parameters
parser.add_argument('--trajectory_index', default=0, type=int, help='trajectory index - set to 0 because we use multiple trajectories')
parser.add_argument('--n_samples', default=5, type=int, help='n_samples')
parser.add_argument('--res', default=8, type=int, help='resolution')
parser.add_argument('--trajectories_num', default=10, type=int, help='number of trajectories to use')
parser.add_argument('--broadcast', default=0, type=int, help='1-integrate the coordinates by broadcasting them as extra dimentions, 2- add coordinates as an extra input')
parser.add_argument('--style', default='brownian', type=str, help='choose syclops style of motion')
parser.add_argument('--loss', default='mean_squared_error', type=str, help='loss type for student')
parser.add_argument('--noise', default=0.15, type=float, help='added noise to the const_p_noise style')
parser.add_argument('--max_length', default=5, type=int, help='choose syclops max trajectory length')


### teacher network parameters
parser.add_argument('--teacher_net', default='/home/orram/Documents/GitHub/imagewalker/teacher_student/model_510046__1628691784.hdf', type=str, help='path to pretrained teacher net')

parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
parser.add_argument('--pd_n_layers', default=1, type=int, help='pd_n_layers')
parser.add_argument('--pd_n_units', default=8, type=int, help='pd_n_units')
parser.add_argument('--pd_d_filter', default=3, type=int, help='pd_d_filter')

parser.add_argument('--student_version', default=3, type=int, help='student version')

parser.add_argument('--last_layer_size', default=128, type=int, help='last_layer_size')


parser.add_argument('--dropout1', default=0.2, type=float, help='dropout1')
parser.add_argument('--dropout2', default=0.0, type=float, help='dropout2')
parser.add_argument('--dataset_norm', default=128.0, type=float, help='dropout2')
parser.add_argument('--dataset_center', dest='dataset_center', action='store_true')
parser.add_argument('--no-dataset_center', dest='dataset_center', action='store_false')

parser.add_argument('--dense_interface', dest='dense_interface', action='store_true')
parser.add_argument('--no-dense_interface', dest='dense_interface', action='store_false')

parser.add_argument('--layer_norm_res', dest='layer_norm_res', action='store_true')
parser.add_argument('--no-layer_norm_res', dest='layer_norm_res', action='store_false')

parser.add_argument('--layer_norm_2', dest='layer_norm_2', action='store_true')
parser.add_argument('--no-layer_norm_2', dest='layer_norm_2', action='store_false')

parser.add_argument('--skip_conn', dest='skip_conn', action='store_true')
parser.add_argument('--no-skip_conn', dest='skip_conn', action='store_false')

parser.add_argument('--last_maxpool_en', dest='last_maxpool_en', action='store_true')
parser.add_argument('--no-last_maxpool_en', dest='last_maxpool_en', action='store_false')


parser.add_argument('--resnet_mode', dest='resnet_mode', action='store_true')
parser.add_argument('--no-resnet_mode', dest='resnet_mode', action='store_false')

parser.add_argument('--nl', default='relu', type=str, help='non linearity')

parser.add_argument('--stopping_patience', default=10, type=int, help='stopping patience')
parser.add_argument('--learning_patience', default=5, type=int, help='stopping patience')
parser.add_argument('--manual_suffix', default='', type=str, help='manual suffix')

parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
parser.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false')

parser.add_argument('--rotation_range', default=0.0, type=float, help='dropout1')
parser.add_argument('--width_shift_range', default=0.1, type=float, help='dropout2')
parser.add_argument('--height_shift_range', default=0.1, type=float, help='dropout2')

##advanced trajectory parameters
parser.add_argument('--time_sec', default=0.3, type=float, help='time for realistic trajectory')
parser.add_argument('--traj_out_scale', default=4.0, type=float, help='scaling to match receptor size')

parser.add_argument('--snellen', dest='snellen', action='store_true')
parser.add_argument('--no-snellen', dest='snellen', action='store_false')

parser.add_argument('--vm_kappa', default=0., type=float, help='factor for emulating sub and super diffusion')


parser.set_defaults(data_augmentation=True,
                    layer_norm_res=True,
                    layer_norm_student=True,
                    batch_norm_student=False,
                    layer_norm_2=True,
                    skip_conn=True,
                    last_maxpool_en=True,
                    testmode=False,
                    dataset_center=True,
                    dense_interface=False,
                    resnet_mode=True,
                    skip_student_training=False,
                    fine_tune_student=False,
                    snellen=True)

config = parser.parse_args()
config = vars(config)
print('config  ',config)

parameters = config
TESTMODE = parameters['testmode']


lsbjob = os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob

# load dataset
if config['n_classes']==10:
    (trainX, trainY), (testX, testY)= cifar10.load_data()
elif config['n_classes']==100:
    (trainX, trainY), (testX, testY) = cifar100.load_data()
else:
    error

images, labels = trainX, trainY

# layer_name = parameters['layer_name']
num_feature = parameters['num_feature']
trajectory_index = parameters['trajectory_index']
n_samples = parameters['n_samples']
res = parameters['res']
trajectories_num = parameters['trajectories_num']
run_index = parameters['run_index']
dropout = parameters['dropout']
rnn_dropout = parameters['rnn_dropout']
this_run_name = parameters['run_name_prefix'] + '_j' + lsbjob + '_t' + str(int(time.time()))
parameters['this_run_name'] = this_run_name
epochs = parameters['epochs']
int_epochs = parameters['int_epochs']
student_block_size = parameters['student_block_size']
print(parameters)
# scale pixels
def prep_pixels(train, test,resnet_mode=False):
    # convert from integers to floats
    if resnet_mode:
        train_norm = cifar10_resnet50.preprocess_image_input(train)
        test_norm = cifar10_resnet50.preprocess_image_input(test)
        print('preprocessing in resnet mode')
    else:
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        #center
        if parameters['dataset_center']:
            mean_image = np.mean(train_norm, axis=0)
            train_norm -= mean_image
            test_norm -= mean_image
        # normalize to range 0-1
        train_norm = train_norm / parameters['dataset_norm']
        test_norm = test_norm /  parameters['dataset_norm']
        # return normalized images
    return train_norm, test_norm

# prepare pixel data
trainX, testX = prep_pixels(trainX, testX, resnet_mode=parameters['resnet_mode'])


#%%
############################### Get Trained Teacher ##########################3

path = os.getcwd() + '/'

save_model_path = path + 'saved_models/{}_feature/'.format(this_run_name)
checkpoint_filepath = save_model_path + '/{}_feature_net_ckpt'.format(this_run_name)
if True:


    #%%
    #################### Get Layer features as a dataset ##########################
    batch_size = 32
    start = 0
    end = batch_size
    train_data = []
    validation_data = []
    upsample_factor = parameters['upsample'] if parameters['upsample'] !=0 else 1
    count = 0

    feature_space = 64
    feature_list = 'all'

    print('\nLoaded feature data from teacher')

    ##################### Define Student #########################################
    verbose =parameters['verbose']

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        mode='min',
        save_best_only=True)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                   cooldown=0,
                                                   patience=5,
                                                   min_lr=0.5e-6)
    early_stopper = keras.callbacks.EarlyStopping(
                                                  min_delta=5e-5,
                                                  patience=10,
                                                  verbose=0,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=True
                                                  )

if parameters['student_version']==3:
    student_fun = student3
elif parameters['student_version'] == 4:
    student_fun = student4
elif parameters['student_version'] == 5:
    student_fun = student5
else:
    error

print('initializing student')


pos_det =pos_det101(sample = parameters['max_length'],
                   res = res,
                    activation = parameters['student_nl'],
                    dropout = dropout,
                    rnn_dropout = rnn_dropout,
                    n_layers= parameters['pd_n_layers'],
                    conv_rnn_type = parameters['conv_rnn_type'],
            n_units= parameters['pd_n_units'],
                    d_filter= parameters['pd_d_filter'],
            loss=parameters['loss'])

pos_det.summary()

train_accur = []
test_accur = []
# generator parameters:

BATCH_SIZE=32
position_dim = (parameters['n_samples'],parameters['res'],parameters['res'],2) if  parameters['broadcast']==1 else (parameters['n_samples'],2)
movie_dim = (parameters['n_samples'], parameters['res'], parameters['res'], 3)
def args_to_dict(**kwargs):
    return kwargs
generator_params = args_to_dict(batch_size=BATCH_SIZE, movie_dim=movie_dim, position_dim=position_dim, n_classes=None, shuffle=True,
                 prep_data_per_batch=True,one_hot_labels=False, one_random_sample=False,
                                    res = parameters['res'],
                                    n_samples = parameters['n_samples'],
                                    mixed_state = True,
                                    n_trajectories = parameters['trajectories_num'],
                                    trajectory_list = 0,
                                    broadcast=parameters['broadcast'],
                                    style = parameters['style'],
                                    max_length=parameters['max_length'],
                                    noise = parameters['noise'],
                                time_sec=parameters['time_sec'], traj_out_scale=parameters['traj_out_scale'],  snellen=parameters['snellen'],vm_kappa=parameters['vm_kappa'])
print('preparing generators')

train_generator_position = Syclopic_dataset_generator(trainX[:-5000], labels[:-5000],return_x1_as_labels=True, **generator_params)
val_generator_position = Syclopic_dataset_generator(trainX[-5000:].repeat(parameters['val_set_mult'],axis=0), labels[-5000:].repeat(parameters['val_set_mult'],axis=0), return_x1_as_labels=True, validation_mode=True, **generator_params)

if True:
    pos_det = keras.models.load_model('saved_models/noname_j_t1636377960_feature/pd_model.hdf')
    # pos_det.evaluate(val_generator_position, verbose = 2)
    xx,yy = val_generator_position[0]
    yy_hat = pos_det.predict(xx)
    print(yy_hat[:5]*232)
    # print(yy.shape)
    print(yy[:5]*232)
    print((yy[:5]-yy_hat[:5])*232)
