from __future__ import division, print_function, absolute_import

import numpy as np
import cv2
import misc
from RL_networks import Stand_alone_net
import pandas as pd
import sys
import matplotlib.pyplot as plt

from keras_utils import *
from misc import *

import importlib
importlib.reload(misc)

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy


import SYCLOP_env as syc


#Define function for low resolution lens on syclop
def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp



import importlib
importlib.reload(misc)
from misc import Logger
import os 

hp = HP()
hp.save_path = 'saved_runs'

hp.description = "syclop cifar check runs "
hp.this_run_name = 'syclop_check'
def deploy_logs():
    if not os.path.exists(hp.save_path):
        os.makedirs(hp.save_path)

    dir_success = False
    for sfx in range(1):  # todo legacy
        candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
        if not os.path.exists(candidate_path):
            hp.this_run_path = candidate_path
            os.makedirs(hp.this_run_path)
            dir_success = True
            break
    if not dir_success:
        error('run name already exists!')

    sys.stdout = Logger(hp.this_run_path+'log.log')
    print('results are in:', hp.this_run_path)
    print('description: ', hp.description)
    #print('hyper-parameters (partial):', hp.dict)




sample = 5#int(sys.argv[4])

cnn_net = 1#int(sys.argv[1]) #Choose CNN one img network

if cnn_net == 1:
    cnn_net = cnn_one_img(n_timesteps = sample, input_size = 32)
elif cnn_net == 2:
    cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = 32)
    

rnn_net = 6#int(sys.argv[2]) #Choose RNN network

if rnn_net == 1:
    rnn_net = rnn_model(n_timesteps = sample, input_size = 32)
elif rnn_net == 2:
    rnn_net = extended_rnn_model(n_timesteps = sample, input_size = 32)
elif rnn_net == 3:
    rnn_net = low_features_rnn_model(n_timesteps = sample, input_size = 32)
elif rnn_net == 4:
    rnn_net = extended_low_features_rnn_model(n_timesteps = sample, input_size = 32)
elif rnn_net == 5:
    rnn_net = no_cnn_low_features_model(input_size = 4, input_dim = 3)
elif rnn_net == 6:
    rnn_net = low_features_middle_cnn_model(input_size = 4, input_dim = 3)    

res = 8#int(sys.argv[3])


    
n_timesteps = sample
def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

#deploy_logs()
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,sample = sample, return_datasets=True, mixed_state = False, add_seed = 0)
train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
# print("##################### Fit {} and trajectories model on training data res = {} ##################".format(cnn_net.name,res))
# cnn_history = cnn_net.fit(
#     train_dataset_x,
#     train_dataset_y,
#     batch_size=64,
#     epochs=30,
#     # We pass some validation for
#     # monitoring validation loss and metrics
#     # at the end of each epoch
#     validation_data=(test_dataset_x, test_dataset_y),
#     verbose = 1)
# print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])
print("##################### Fit {} and trajectories model on training data res = {} ##################".format(rnn_net.name,res))
rnn_history = rnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=30,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 1)
print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])