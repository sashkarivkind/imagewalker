#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Looking at different network architectures to see if they bring different results
for res=6 or other. 
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import scipy.stats as st
import sys 

#import torch
#from torch.optim import Adam, SGD
#import torch.nn as nn

import tensorflow as tf
import tensorflow.keras as keras

from mnist import MNIST

from keras_utils import * 

print('Starting Run')

#DEfine the number of trajectories to use
num_trajectories = int(sys.argv[1])

num_learning_epochs = int(sys.argv[2])

num_trials = int(sys.argv[3])

res = int(sys.argv[4])

sample = int(sys.argv[5]) #Number of samples to drew
['rnn_model','rnn_model_dense','simple_vanila_model']

print('num_trajectories = ', num_trajectories, 'num_learning_epochs = ',num_learning_epochs, 'num_trials', num_trials, 'res', res)

#define the place holders to hold the detials of each run 
#One dataframe for the RNN eith the coordinates insertes
#columns_train = []
#columns_test = []
#columns_test_no_ccor
#for trial in range(num_trials):
#    columns_train.append('trial_{}_train_loss'.format(trial))
#    columns_test.append('trial_{}_test_accur'.format(trial))
#    columns_test_no_ccor.append('trial_{}_no_coord_test_accur'.format(trial))
    
    
train_dataframe = pd.DataFrame()
test_dataframe = pd.DataFrame()
test_no_coor_dataframe = pd.DataFrame()
test_regular_rnn_dataframe = pd.DataFrame()
#The second dataframe is for the RNN without the coordinates
columns = []

for trial in range(num_trials):
    columns.append('trial_{}_train_loss'.format(trial))
    columns.append('trial_{}_test_accur'.format(trial))
    
train_dataframe_no_coordinates = pd.DataFrame()
test_dataframe_no_coordinates = pd.DataFrame()



mnist = MNIST('/home/labs/ahissarlab/orra/datasets/mnist')
images, labels = mnist.load_training()

n_timesteps = 5


def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1)[...,np.newaxis],np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)


def train(net, num_learning_epochs, dataframe,
          train_dataset_x,train_dataset_y,test_dataset_x,test_dataset_y):
    
    print("##########Fit {} and trajectories model on training data######".format(net.name))
    history = net.fit(
        train_dataset_x,
        train_dataset_y,
        batch_size=64,
        epochs=num_learning_epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(test_dataset_x, test_dataset_y),
        verbose = 0) #(validation_images, validation_labels)
    print(net.name, 'Validation Accuracy = ',history.history['val_sparse_categorical_accuracy'])
    results = net.evaluate((test_dataset_x[0],test_dataset_x[1]*np.mean(test_dataset_x[1])), test_dataset_y, verbose = 0)
    print('When avarging out the trajectories test accuracy = ', results[1])
    dataframe[net.name] = history.history['val_sparse_categorical_accuracy']
    return history.history['val_sparse_categorical_accuracy'], dataframe

net_dataframe = pd.DataFrame()    
    
    
for trial in range(num_trials):
    print('###################### Trial number {} ###########################'.format(trial))
    
        
    train_dataset, test_dataset = create_mnist_dataset(images, labels,res = res,sample = sample,return_datasets=True, add_seed = num_trajectories)
    train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
    hist, net_dataframe = train(net = rnn_model(), num_learning_epochs = num_learning_epochs, 
                                dataframe = net_dataframe,
                    train_dataset_x = train_dataset_x,
                    train_dataset_y = train_dataset_y,
                    test_dataset_x = test_dataset_x,
                    test_dataset_y = test_dataset_y)
    
    hist, net_dataframe = train(net = rnn_model_dense(), num_learning_epochs = num_learning_epochs, 
                            dataframe = net_dataframe,
                train_dataset_x = train_dataset_x,
                train_dataset_y = train_dataset_y,
                test_dataset_x = test_dataset_x,
                test_dataset_y = test_dataset_y)
    
    hist, net_dataframe = train(net = rnn_model_concat_same_length(), num_learning_epochs = num_learning_epochs, 
                        dataframe = net_dataframe,
            train_dataset_x = train_dataset_x,
            train_dataset_y = train_dataset_y,
            test_dataset_x = test_dataset_x,
            test_dataset_y = test_dataset_y)
        
    hist, net_dataframe = train(net = simple_rnn_model(), num_learning_epochs = num_learning_epochs, 
                    dataframe = net_dataframe,
        train_dataset_x = train_dataset_x,
        train_dataset_y = train_dataset_y,
        test_dataset_x = test_dataset_x,
        test_dataset_y = test_dataset_y)
            
            
    hist, net_dataframe = train(net = simple_vanila_model(), num_learning_epochs = num_learning_epochs, 
                dataframe = net_dataframe,
    train_dataset_x = train_dataset_x,
    train_dataset_y = train_dataset_y,
    test_dataset_x = test_dataset_x,
    test_dataset_y = test_dataset_y)

#########################   Save Data   ###################################### 
net_dataframe.to_pickle('net_compare_dataframe')
    
       
print(net_dataframe.iloc[-1,:])
