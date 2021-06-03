
from __future__ import division, print_function, absolute_import

import numpy as np
import cv2
import misc
from RL_networks import Stand_alone_net
import pickle
import pandas as pd
import sys

import tensorflow.keras as keras
from keras_utils import *

from sklearn.metrics import confusion_matrix

from tensorflow.keras.datasets import cifar10
import tensorflow as tf
# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()

import importlib
importlib.reload(misc)


images, labels = trainX, trainy

import matplotlib.pyplot as plt
import SYCLOP_env as syc

epochs = int(sys.argv[1])
hidden_size = int(sys.argv[2])




#Define function for low resolution lens on syclop
def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp

import importlib
importlib.reload(misc)

n_timesteps = 5

def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

rnn_memory = pd.DataFrame()
simple_rnn_memory = pd.DataFrame()
cnn_memory = pd.DataFrame()

simple_rnn_accuracy = []
rnn_accuracy = []
cnn_accuracy = []

res_list = [25,10,8]
print('00000')
for res in res_list:
    print('############################################## RES = {} #############################################'.format(res))
    train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,return_datasets=True, mixed_state = False, add_seed = 0)
    train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
    
    # print("####################### Fit RNN model on training data ##############################")
    # simple_rnn_net = simple_rnn_model()
    # simple_rnn_history = simple_rnn_net.fit(
    #     train_dataset_x,
    #     train_dataset_y,
    #     batch_size=64,
    #     epochs=epochs,
    #     # We pass some validation for
    #     # monitoring validation loss and metrics
    #     # at the end of each epoch
    #     validation_data=(test_dataset_x, test_dataset_y),
    #     verbose = 0) #(validation_images, validation_labels)
    # print('################# RNN Validation Accuracy = ',simple_rnn_history.history['val_sparse_categorical_accuracy'])
    # simple_rnn_memory[res] = simple_rnn_history.history['val_sparse_categorical_accuracy']
    # simple_rnn_accuracy.append(max(simple_rnn_history.history['val_sparse_categorical_accuracy']))
    print("####################### Fit RNN and trajectories model on training data ##############################")
    rnn_net = rnn_model(input_size = 32)
    rnn_history = rnn_net.fit(
        train_dataset_x,
        train_dataset_y,
        batch_size=64,
        epochs=epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(test_dataset_x, test_dataset_y),
        verbose = 0) #(validation_images, validation_labels)
    print('################# Trajectories RNN Combined Validation Accuracy = ',rnn_history.history['val_sparse_categorical_accuracy'])
    print('################# Trajectories RNN Combined Train Accuracy = ',rnn_history.history['sparse_categorical_accuracy'])
    rnn_memory[res] = rnn_history.history['val_sparse_categorical_accuracy']
    rnn_accuracy.append(max(rnn_history.history['val_sparse_categorical_accuracy']))
    print("####################### Fit CNN and trajectories model on training data ##############################")
    cnn_net = cnn_one_img(input_size = 32)
    cnn_history = cnn_net.fit(
        train_dataset_x,
        train_dataset_y,
        batch_size = 64,
        epochs = epochs,
        validation_data = (test_dataset_x, test_dataset_y),
        verbose = 0)
    print('################# CNN Validation Accuracy = ',cnn_history.history['val_sparse_categorical_accuracy'])
    print('################# CNN Train Accuracy = ',cnn_history.history['sparse_categorical_accuracy'])
    cnn_memory[res] = cnn_history.history['val_sparse_categorical_accuracy']
    cnn_accuracy.append(max(cnn_history.history['val_sparse_categorical_accuracy']))
   

rnn_memory.to_pickle('RNN_cifar_test_accuracy_{}'.format(hidden_size))
rnn_memory.to_pickle('simple_RNN_cifar_test_accuracy_{}'.format(hidden_size))
cnn_memory.to_pickle('CNN_cifar_test_accuracy_{}'.format(hidden_size))
print(rnn_accuracy, cnn_accuracy)
x = 32/np.array(res_list)
plt.figure()
plt.plot(x, rnn_accuracy,'^', label = 'RNN test' )
plt.plot(x, simple_rnn_accuracy, 'H',label = 'simple RNN test')
plt.plot(x, cnn_accuracy, 'o',label = 'CNN one image test')
plt.legend()
plt.savefig('syclop_cifar_101_accuracies_h={}_ep={}.png'.format(hidden_size, epochs))



