'''
The follwing code runs a test RNN with a dense layer to integrate the coordinates
network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 

Running a net [32,64,128] only ConvLSTM without dropout
################# ConvLSTM_False Validation Accuracy =  [0.4234, 0.4724, 0.4916, 0.5356, 0.5498, 0.548, 0.5452, 0.5408, 0.5452, 0.5444, 0.536, 0.5438, 0.5256, 0.533, 0.545, 0.5322, 0.5288, 0.5406, 0.5374, 0.5382, 0.5262, 0.5358, 0.5286, 0.529, 0.539, 0.5346, 0.5432, 0.5398, 0.5298, 0.5296]
################# ConvLSTM_False Training Accuracy =  [0.352, 0.4515111, 0.51217777, 0.5662, 0.63375556, 0.7343111, 0.8472, 0.92388886, 0.95086664, 0.9685111, 0.9725111, 0.97726667, 0.97882223, 0.9786889, 0.98053336, 0.98477775, 0.98322225, 0.9847556, 0.98584443, 0.9860889, 0.9863333, 0.9879778, 0.98895556, 0.9884, 0.9898889, 0.98928887, 0.9894222, 0.99126667, 0.99175555, 0.9908]
Running a net [32,64,128] only ConvLSTM with dropout = 0.2


The same with no dropout:



'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import sys
sys.path.insert(1, "/home/orram/Documents/GitHub/imagewalker/")#'/home/labs/ahissarlab/orra/imagewalker')


import numpy as np
import cv2
import misc
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from keras_utils import *
from misc import *

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy


#Define function for low resolution lens on syclop
def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp

def bad_res102(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_AREA)
    return dwnsmp

import importlib
importlib.reload(misc)
from misc import Logger
import os 


def deploy_logs():
    if not os.path.exists(hp.save_path):
        os.makedirs(hp.save_path)

    dir_success = False
    for sfx in range(1):  # todo legacy
        candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
        if not os.path.exists(candidate_path):
            hp.this_run_path = candidate_path
            os.makedirs(hp.this_run_path)
            dir_success = Truecnn_net = cnn_one_img(n_timesteps = sample, input_size = 28, input_dim = 1)
            break
    if not dir_success:
        error('run name already exists!')

    sys.stdout = Logger(hp.this_run_path+'log.log')
    print('results are in:', hp.this_run_path)
    print('description: ', hp.description)
    #print('hyper-parameters (partial):', hp.dict)

epochs = 10#int(sys.argv[1])

sample = 5#int(sys.argv[2])

res = 8#int(sys.argv[3])

hidden_size = 128#int(sys.argv[4])
   
cnn_dropout = 0.3

rnn_dropout = 0.2

n_timesteps = sample
def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def convgru(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 3, concat = False):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    
    # define LSTM model
    x = keras.layers.ConvLSTM2D(32, 3, return_sequences=True, padding = 'same')(inputA)
    #x = keras.layers.ConvLSTM2D(32, 3, return_sequences=True, padding = 'valid')(x)
    print(x.shape)
    #x = keras.layers.ConvLSTM2D(64, 2, return_sequences=True, padding = 'same')(x)
    x = keras.layers.ConvLSTM2D(64, 2, return_sequences=True, padding = 'valid')(x)
    print(x.shape)
    #x = keras.layers.ConvLSTM2D(128, 2, return_sequences=True, padding = 'same')(x)
    x = keras.layers.ConvLSTM2D(128, 2, return_sequences=True, padding = 'valid')(x)
    print(x.shape)
    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
    print(x.shape)
    if concat:
        x = keras.layers.Concatenate()([x,inputB])
    print(x.shape)
    x = keras.layers.Flatten()(x)
    print(x.shape)
    x = keras.layers.Dense(1024,activation="relu")(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    print(x.shape)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'ConvLSTM_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = convgru(n_timesteps = sample, cell_size = hidden_size,input_size = 8  , concat = True)
cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = 8)

# hp = HP()
# hp.save_path = 'saved_runs'

# hp.description = "syclop cifar net search runs"
# hp.this_run_name = 'syclop_{}'.format(rnn_net.name)
# deploy_logs()
#%%
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )#bad_res_func = bad_res101, up_sample = True)

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
#%%
print("##################### Fit {} and trajectories model on training data res = {} ##################".format(cnn_net.name,res))
cnn_history = cnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 1)
print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])

#%%
print("##################### Fit {} and trajectories model on training data res = {} ##################".format(rnn_net.name,res))
rnn_history = rnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 1)

#print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])
#print('################# {} Training Accuracy = '.format(cnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
plt.legend()
plt.title('{} on cifar res = {} hs = {} dropout = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout))
plt.savefig('{} on Cifar res = {} val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict{}_{}'.format(rnn_net.name, hidden_size,cnn_dropout), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict'.format(cnn_net.name), 'wb') as file_pi:
    pickle.dump(cnn_history.history, file_pi)