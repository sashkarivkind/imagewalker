#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
added the explore relations part after 735561
"""

import os
import sys
import gc
sys.path.insert(1, '/home/labs/ahissarlab/arivkind/imagewalker')
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import time
import pickle
import argparse
# from feature_learning_utils import  student3,  write_to_file, traject_learning_dataset_update,  net_weights_reinitializer
from keras_utils import create_cifar_dataset, split_dataset_xy
from dataset_utils import Syclopic_dataset_generator, test_num_of_trajectories
print(os.getcwd() + '/')
#%%

parser = argparse.ArgumentParser()

#general parameters
parser.add_argument('--run_name_prefix', default='noname', type=str, help='path to pretrained teacher net')
parser.add_argument('--run_index', default=10, type=int, help='run_index')

parser.add_argument('--testmode', dest='testmode', action='store_true')
parser.add_argument('--no-testmode', dest='testmode', action='store_false')

### student parameters
parser.add_argument('--epochs', default=1, type=int, help='num training epochs')
parser.add_argument('--int_epochs', default=1, type=int, help='num internal training epochs')
parser.add_argument('--decoder_epochs', default=20, type=int, help='num internal training epochs')
parser.add_argument('--num_feature', default=64, type=int, help='legacy to be discarded')
parser.add_argument('--rnn_layer1', default=32, type=int, help='legacy to be discarded')
parser.add_argument('--rnn_layer2', default=64, type=int, help='legacy to be discarded')
parser.add_argument('--time_pool', default=0, help='time dimention pooling to use - max_pool, average_pool, 0')

parser.add_argument('--student_block_size', default=1, type=int, help='number of repetition of each convlstm block')
parser.add_argument('--conv_rnn_type', default='lstm', type=str, help='conv_rnn_type')
parser.add_argument('--student_nl', default='relu', type=str, help='non linearity')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout1')
parser.add_argument('--rnn_dropout', default=0.0, type=float, help='dropout1')
conv_rnn_type='lstm'

parser.add_argument('--layer_norm_student', dest='layer_norm_student', action='store_true')
parser.add_argument('--no-layer_norm_student', dest='layer_norm_student', action='store_false')


### syclop parameters
parser.add_argument('--trajectory_index', default=0, type=int, help='trajectory index - set to 0 because we use multiple trajectories')
parser.add_argument('--n_samples', default=5, type=int, help='sample')
parser.add_argument('--res', default=8, type=int, help='resolution')
parser.add_argument('--trajectories_num', default=-1, type=int, help='number of trajectories to use')
parser.add_argument('--broadcast', default=1, type=int, help='1-integrate the coordinates by broadcasting them as extra dimentions, 2- add coordinates as an extra input')
parser.add_argument('--style', default='spiral_2dir2', type=str, help='choose syclops style of motion')
# parser.add_argument('--style', default='spiral', type=str, help='choose syclops style of motion')
# parser.add_argument('--style', default='degenerate_fix2', type=str, help='choose syclops style of motion')
# parser.add_argument('--style', default='xx1_intoy_rucci', type=str, help='choose syclops style of motion')
# parser.add_argument('--style', default='xx1_vonmises_walk', type=str, help='choose syclops style of motion')
# parser.add_argument('--style', default='ZigZag', type=str, help='choose syclops style of motion')
# parser.add_argument('--style', default='const_p_noise', type=str, help='choose syclops style of motion')
# parser.add_argument('--style', default='spiral_2dir_shfl', type=str, help='choose syclops style of motion')
# parser.add_argument('--style', default='brownian', type=str, help='choose syclops style of motion')
parser.add_argument('--noise', default=0.5, type=float, help='added noise to the const_p_noise style')

parser.add_argument('--max_length', default=5, type=int, help='choose syclops max trajectory length')
parser.add_argument('--val_set_mult', default=2, type=int, help='repetitions of validation dataset to reduce trajectory noise')

##advanced trajectory parameters
parser.add_argument('--time_sec', default=0.3, type=float, help='time for realistic trajectory')
parser.add_argument('--traj_out_scale', default=4.0, type=float, help='scaling to match receptor size')

parser.add_argument('--snellen', dest='snellen', action='store_true')
parser.add_argument('--no-snellen', dest='snellen', action='store_false')

parser.add_argument('--vm_kappa', default=-20., type=float, help='factor for emulating sub and super diffusion')




### teacher network parameters
parser.add_argument('--teacher_net', default='/home/orram/Documents/GitHub/imagewalker/teacher_student/model_510046__1628691784.hdf', type=str, help='path to pretrained teacher net')

parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
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

parser.add_argument('--nl', default='relu', type=str, help='non linearity')

parser.add_argument('--stopping_patience', default=10, type=int, help='stopping patience')
parser.add_argument('--learning_patience', default=5, type=int, help='stopping patience')
parser.add_argument('--manual_suffix', default='', type=str, help='manual suffix')

parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
parser.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false')

parser.add_argument('--rotation_range', default=0.0, type=float, help='dropout1')
parser.add_argument('--width_shift_range', default=0.1, type=float, help='dropout2')
parser.add_argument('--height_shift_range', default=0.1, type=float, help='dropout2')

parser.set_defaults(data_augmentation=True,layer_norm_res=True,layer_norm_student=True,layer_norm_2=True,skip_conn=True,last_maxpool_en=True, testmode=False,dataset_center=True, dense_interface=False,
                    snellen=False)

config = parser.parse_args()
config = vars(config)
print('config  ',config)
parameters=config
# load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
images, labels = trainX, trainY
BATCH_SIZE=32
position_dim = (parameters['n_samples'],parameters['res'],parameters['res'],2) if  parameters['broadcast']==1 else (parameters['n_samples'],2)
def args_to_dict(**kwargs):
    return kwargs
generator_params = args_to_dict(batch_size=BATCH_SIZE, movie_dim=(parameters['n_samples'],parameters['res'],parameters['res'],3), position_dim=position_dim, n_classes=None, shuffle=True,
                 prep_data_per_batch=True,one_hot_labels=False,
                                    res = parameters['res'],
                                    n_samples = parameters['n_samples'],
                                    mixed_state = True,
                                    n_trajectories = parameters['trajectories_num'],
                                    trajectory_list = 0,
                                    broadcast=parameters['broadcast'],
                                    style = parameters['style'],
                                    max_length=parameters['max_length'],
                                    noise = parameters['noise'],
                                    time_sec=parameters['time_sec'], traj_out_scale=parameters['traj_out_scale'],  snellen=parameters['snellen'],vm_kappa=parameters['vm_kappa']
                                )
train_generator = Syclopic_dataset_generator(images[:-5000], labels[:-5000], **generator_params)
val_generator = Syclopic_dataset_generator(images[-5000:].repeat(parameters['val_set_mult'],axis=0), labels[-5000:].repeat(parameters['val_set_mult'],axis=0), validation_mode=True, **generator_params)

train_generator_pic = Syclopic_dataset_generator(images[:5000], images[:5000,:8,:8,:]+1, **generator_params)
val_generator_pic = Syclopic_dataset_generator(images[-5000:], images[-5000:,:8,:8,:]+1, validation_mode=True, **generator_params)
#
train_generator_rndsmpl = Syclopic_dataset_generator(images[:5000], labels[:5000], one_random_sample=True, **generator_params)
val_generator_rndsmpl = Syclopic_dataset_generator(images[-5000:], labels[-5000:], one_random_sample=True, validation_mode=True, **generator_params)
# train_generator_rndsmpl = Syclopic_dataset_generator(images[:5000], labels[:5000],**generator_params)
# val_generator_rndsmpl = Syclopic_dataset_generator(images[-5000:], labels[-5000:],  one_random_sample=True, **generator_params)

inputA = keras.layers.Input(shape=( parameters['n_samples'], parameters['res'],parameters['res'],3))
# inputB = keras.layers.Input(shape=( parameters['n_samples'],parameters['res'],parameters['res'],2))
if parameters['broadcast']==1:
    inputB = keras.layers.Input(shape=( parameters['n_samples'], parameters['res'],parameters['res'],2))
    print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
else:
    inputB = keras.layers.Input(shape=( parameters['n_samples'],2))
x = keras.layers.Flatten()(inputA)
x = keras.layers.Dense(10, activation="softmax",
                       name='final')(x)
model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'student_3')
opt=tf.keras.optimizers.Adam(lr=1e-3)
model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# inputC = keras.layers.Input(shape=( parameters['n_samples'], parameters['res'],parameters['res'],3))

x = tf.keras.layers.AveragePooling3D(pool_size=( parameters['n_samples'], 1, 1))(inputA)
x = tf.squeeze(x,1)

model2 = keras.models.Model(inputs=[inputA],outputs=x, name = 'student_3')
# model2 = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'student_3')
opt=tf.keras.optimizers.Adam(lr=1e-3)
model2.compile(
    optimizer=opt,
    loss="mean_squared_error",
    metrics=["mean_squared_error"],
)
model2.summary()

input3 = keras.layers.Input(shape=(parameters['res'],parameters['res'],3))
x = keras.layers.Flatten()(input3)
x = keras.layers.Dense(10, activation="softmax",
                       name='final')(x)
model3 = keras.models.Model(inputs=[input3],outputs=x, name = 'student_3')
opt=tf.keras.optimizers.Adam(lr=1e-3)
model3.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model3.summary()


# model.fit_generator(train_generator,  validation_data=val_generator, epochs=5, workers=8, use_multiprocessing=True)
# ppp=model(val_generator[0])
# print('---------')
# def test_num_of_trajectories(gen,batch_size=32):
#     zz=[]
#     cc=0
#     for uu in range(len(gen)):
#         for bb in range(batch_size):
#             zz.append(str(gen[uu][0][1][bb, :, 0, 0, :]))
#             cc += 1
#     return len(set(zz)), cc
# print('---------')
# print('-------- total trajectories {}, out of tries: {}'.format( *test_num_of_trajectories(val_generator)))
# print('-------- total trajectories {}, out of tries: {}'.format( *test_num_of_trajectories(train_generator)))

# print('prediction shape',ppp.shape )
# print('val_generator len',len(val_generator))
# print('evaluating train set')
# for ii in range(10):
#     model.evaluate(train_generator, workers=8, use_multiprocessing=True)
# print('evaluating validation set')
# for ii in range(10):
#     model.evaluate(val_generator, workers=8, use_multiprocessing=True)
#
# model2.fit_generator(train_generator_pic,  validation_data=val_generator_pic, epochs=5, workers=8, use_multiprocessing=True)
model3.fit_generator(train_generator_rndsmpl,  validation_data=val_generator_rndsmpl, epochs=0, workers=8, use_multiprocessing=True)

print(val_generator_rndsmpl[0][0])

# import matplotlib.pyplot as plt
# zz = []
# cc = 0
# plt.figure()
# for uu in range(3):
#     for bb in range(3):
#         traj=val_generator[uu][0][1][bb, :, 0, 0, :]
#         cc += 1
#         plt.subplot(3,3,bb*3+uu+1)
#         plt.plot(traj[:,0],traj[:,1],'-o')
#         plt.title(parameters['style'])
# plt.show()