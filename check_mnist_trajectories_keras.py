#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checking the mixing of trajectories - limits of number of trajectories and 
the limits on running the network for long time. Plus, the contribution 
of inserting the trajectories to the network.

For each number of trajectories (5, 20, 50, 100, 200, 1000, inf)

For each RNN (with or without trajectories)

Run for 200 epochs

Run a control of the trajectories where you don't input the trajectories'
Save test_accuracy for each run 

Save a dataset with the final accuracy from the trajectories

Print fig of with/without trajectories

sys.argv gets:
    
[1] = how many trajectories
[2] = number of epochs per run
[3] = number of epochs
[4] = resolution factor

"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import scipy.stats as st
import sys 
import gc

#import torch
#from torch.optim import Adam, SGD
#import torch.nn as nn

import tensorflow as tf
import tensorflow.keras as keras

from mnist import MNIST

from keras_utils import * 

print('Starting Run')

#DEfine the number of trajectories to use
num_trajectories = 1#int(sys.argv[1])

num_learning_epochs = 1# int(sys.argv[2])

num_trials = 1#int(sys.argv[3])

res = 6#int(sys.argv[4])

sample = 5# int(sys.argv[5]) #Number of samples to drew
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



for trial in range(num_trials):
    print('###################### Trial number {} ###########################'.format(trial))
    print("Combined RNN + Trajectories")
    net = rnn_model(n_timesteps = sample, input_dim = 1)
    train_accuracy = []
    test_accuracy = []
    test_no_coor_accuracy = []
    
    print("Regular RNN")
    rnn_net = simple_rnn_model(n_timesteps = sample)
    train_accuracy_rnn = []
    test_accuracy_rnn = []
    test_no_coor_accuracy_rnn = []
    for epep in range(num_learning_epochs):
        
        train_dataset, test_dataset = create_mnist_dataset(images, labels,res = res,sample = sample,return_datasets=True, add_seed = num_trajectories)
        train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
        test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
        del train_dataset
        del test_dataset
        gc.collect()
        print("##########Fit RNN and trajectories model on training data######")
        history = net.fit(
            train_dataset_x,
            train_dataset_y,
            batch_size=64,
            epochs=1,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(test_dataset_x, test_dataset_y),
            verbose = 0) #(validation_images, validation_labels)
        print('Combined RNN, epoch = ',epep, 'Validation Accuracy = ',history.history['val_sparse_categorical_accuracy'][0])
        train_accuracy.append(history.history['sparse_categorical_accuracy'][0])
        test_accuracy.append(history.history['val_sparse_categorical_accuracy'][0])
        results = net.evaluate((test_dataset_x[0],test_dataset_x[1]*0), test_dataset_y, verbose = 0)
        print('When zeroing out the trajectories test accuracy = ', results[1])
        test_no_coor_accuracy.append(results[1])
        
        print("##################Fit RNN model on training data##############")
        history2 = rnn_net.fit(
            train_dataset_x,
            train_dataset_y,
            batch_size=64,
            epochs=1,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(test_dataset_x, test_dataset_y),
            verbose = 0) #(validation_images, validation_labels)
        
        print('Regular RNN, epoch = ',epep, 'Validation Accuracy = ',history2.history['val_sparse_categorical_accuracy'][0])
        
        results2 = rnn_net.evaluate((test_dataset_x[0],test_dataset_x[1]*0), test_dataset_y, verbose = 0)
        train_accuracy_rnn.append(history2.history['sparse_categorical_accuracy'][0])
        test_accuracy_rnn.append(history2.history['val_sparse_categorical_accuracy'][0])
        print('When zeroing out the trajectories test accuracy = ', results2[1])
        test_no_coor_accuracy_rnn.append(results2[1])
    train_dataframe['trial_{}_train_loss'.format(trial)] = train_accuracy
    test_dataframe['trial_{}_test_accur'.format(trial)] = test_accuracy
    test_no_coor_dataframe['trial_{}_no_coord_test_accur'.format(trial)]  = test_no_coor_accuracy
    test_regular_rnn_dataframe['trial_{}_no_coord_test_accur'.format(trial)]  = test_no_coor_accuracy
    train_dataframe_no_coordinates['trial_{}_train_loss'.format(trial)] = train_accuracy_rnn
    test_dataframe_no_coordinates['trial_{}_test_accur'.format(trial)] = test_accuracy_rnn

#########################   Save Data   ######################################
train_dataframe.to_pickle('train_dataset_{}_{}'.format(num_trajectories,sample))
test_dataframe.to_pickle('test_dataset_{}_{}'.format(num_trajectories,sample))
test_no_coor_dataframe.to_pickle('test_no_coor_dataset{}_{}'.format(num_trajectories,sample))
train_dataframe_no_coordinates.to_pickle('train_dataset_no_coordinates{}_{}'.format(num_trajectories,sample))
test_dataframe_no_coordinates.to_pickle('test_dataset_no_coordinates{}_{}'.format(num_trajectories,sample))
#########################   Plot Data    ######################################

#Plot train
train_dataframe['mean'] = train_dataframe.mean(numeric_only = True, axis = 1)
train_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(train_dataframe) - 1, loc = train_dataframe.mean(axis = 1), scale = st.sem(train_dataframe, axis = 1))[0]
train_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(train_dataframe) - 1, loc = train_dataframe.mean(axis = 1), scale = st.sem(train_dataframe, axis = 1))[1]
plt.figure()
x = np.arange(len(train_dataframe))
y = train_dataframe['mean']
plt.plot(x, y,'r',label = 'with coordinates')
plt.fill_between(x, train_dataframe['confidance-'] , train_dataframe['confidance+'])
plt.title('Mean train accuracy from {} trials \n with {} reandom trajectories'.format(num_trials, num_trajectories))
plt.savefig('train_accuracy_{}.png'.format(num_trajectories))

train_dataframe_no_coordinates['mean'] = train_dataframe_no_coordinates.mean(numeric_only = True, axis = 1)
train_dataframe_no_coordinates['confidance-'] = st.t.interval(alpha = 0.95, df = len(train_dataframe_no_coordinates) - 1, loc = train_dataframe_no_coordinates.mean(axis = 1), scale = st.sem(train_dataframe_no_coordinates, axis = 1))[0]
train_dataframe_no_coordinates['confidance+'] = st.t.interval(alpha = 0.95, df = len(train_dataframe_no_coordinates) - 1, loc = train_dataframe_no_coordinates.mean(axis = 1), scale = st.sem(train_dataframe_no_coordinates, axis = 1))[1]
x = np.arange(len(train_dataframe_no_coordinates))
y = train_dataframe_no_coordinates['mean']
plt.plot(x, y,'g',label = 'without coordinates')
plt.fill_between(x, train_dataframe_no_coordinates['confidance-'] , train_dataframe_no_coordinates['confidance+'])
plt.title('Mean train accuracy from {} trials \n with {} reandom trajectories'.format(num_trials, num_trajectories))
plt.legend()
plt.savefig('train_accuracy_{}.png'.format(num_trajectories))

#Plot Test
test_dataframe['mean'] = test_dataframe.mean(numeric_only = True, axis = 1)
test_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_dataframe) - 1, loc = test_dataframe.mean(axis = 1), scale = st.sem(test_dataframe, axis = 1))[0]
test_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_dataframe) - 1, loc = test_dataframe.mean(axis = 1), scale = st.sem(test_dataframe, axis = 1))[1]
plt.figure()
x = np.arange(len(test_dataframe))
y = test_dataframe['mean']
plt.plot(x, y, 'r',label = 'with coordinates')
plt.fill_between(x, test_dataframe['confidance-'] , test_dataframe['confidance+'],alpha = 0.4)

test_no_coor_dataframe['mean'] = test_no_coor_dataframe.mean(numeric_only = True, axis = 1)
test_no_coor_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_no_coor_dataframe) - 1, loc = test_no_coor_dataframe.mean(axis = 1), scale = st.sem(test_no_coor_dataframe, axis = 1))[0]
test_no_coor_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_no_coor_dataframe) - 1, loc = test_no_coor_dataframe.mean(axis = 1), scale = st.sem(test_no_coor_dataframe, axis = 1))[1]
x = np.arange(len(test_no_coor_dataframe))
y = test_no_coor_dataframe['mean']
plt.plot(x, y, 'g',label = 'without coordinates')
plt.fill_between(x, test_no_coor_dataframe['confidance-'] , test_no_coor_dataframe['confidance+'], alpha = 0.4)

test_dataframe_no_coordinates['mean'] = test_dataframe_no_coordinates.mean(numeric_only = True, axis = 1)
test_dataframe_no_coordinates['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_dataframe_no_coordinates) - 1, loc = test_dataframe_no_coordinates.mean(axis = 1), scale = st.sem(test_dataframe_no_coordinates, axis = 1))[0]
test_dataframe_no_coordinates['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_dataframe_no_coordinates) - 1, loc = test_dataframe_no_coordinates.mean(axis = 1), scale = st.sem(test_dataframe_no_coordinates, axis = 1))[1]
x = np.arange(len(test_dataframe_no_coordinates))
y = test_dataframe_no_coordinates['mean']
plt.plot(x, y,label = 'Regular RNN no coordinates integration')
plt.fill_between(x, test_dataframe_no_coordinates['confidance-'] , test_dataframe_no_coordinates['confidance+'],alpha = 0.4)

plt.title('Mean test accuracy from {} trials \n with {} reandom trajectories'.format(num_trials, num_trajectories))
plt.legend()
plt.savefig('test_accuracy_{}.png'.format(num_trajectories))

#########################   Save Data   ######################################
train_dataframe.to_pickle('train_dataset_{}_{}'.format(num_trajectories,sample))
test_dataframe.to_pickle('test_dataset_{}_{}'.format(num_trajectories,sample))
test_no_coor_dataframe.to_pickle('test_no_coor_dataset{}_{}'.format(num_trajectories,sample))
train_dataframe_no_coordinates.to_pickle('train_dataset_no_coordinates{}_{}'.format(num_trajectories,sample))
test_dataframe_no_coordinates.to_pickle('test_dataset_no_coordinates{}_{}'.format(num_trajectories,sample))

