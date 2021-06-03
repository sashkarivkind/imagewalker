'''
The follwing code runs a test RNN with a dense layer to integrate the coordinates
network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 

Running an all convLSTM network with [32,32,64,64,128,128] dropout = 0.1 learnt nothing
ConvLSTM_True Validation Accuracy =  [0.0976, 0.1058, 0.0958, 0.1038, 0.1038, 0.1024, 0.097, 0.1024, 0.0976, 0.1058, 0.0976, 0.0986, 0.095, 0.0958, 0.097, 0.1058, 0.1038, 0.0976, 0.0958, 0.0958, 0.0976, 0.0976, 0.1064, 0.0976, 0.1038, 0.0986, 0.0986, 0.0986, 0.1024, 0.0958]
ConvLSTM_True Training Accuracy =  [0.09735555, 0.09991111, 0.10057778, 0.098177776, 0.099022225, 0.09866667, 0.09928889, 0.098844446, 0.097333334, 0.09877778, 0.099244446, 0.09762222, 0.09753333, 0.1006, 0.09808889, 0.0986, 0.10008889, 0.09957778, 0.09866667, 0.09933333, 0.098733336, 0.09802222, 0.09848889, 0.098688886, 0.10022222, 0.097666666, 0.101111114, 0.101222225, 0.10253333, 0.09933333]

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
    return dwnsmpres = 6# int(sys.argv[3])

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
            dir_success = Truecnn_net = cnn_one_img(res = 6# int(sys.argv[3])n_timesteps = sample, input_size = 28, input_dim = 1)
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

def convgru(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 1, concat = False):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define LSTM model
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(inputA)
    print(x.shape)
    # x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(x)
    # print(x.shape)
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(x)
    print(x.shape)
    x = keras.layers.Flatten()(x)
    print(x.shape)
    if concat:
        x = keras.layers.Concatenate()([x,inputB])
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'Basic_ConvLSTM_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = convgru(n_timesteps = sample, cell_size = hidden_size,input_size = 8, input_dim = 3  , concat = False)
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