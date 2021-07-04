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
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')

import gc

#import torch
#from torch.optim import Adam, SGD
#import torch.nn as nn

# import tensorflow as tf
# import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy


from keras_utils import create_cifar_dataset, split_dataset_xy
from cifar_nets import cnn_gru, parallel_gru

print('Starting Run')

nets = [cnn_gru, parallel_gru]
if len(sys.argv) > 1:
    parameters = {
    #DEfine the number of trajectories to use
        'num_trajectories' : int(sys.argv[1]),
    
        'num_learning_epochs' : int(sys.argv[2]),
    
        'num_trials' : int(sys.argv[3]),
    
        'res' : int(sys.argv[4]),
    
        'sample' : int(sys.argv[5]),  #Number of samples to drew,
        
        'net' : nets[int(sys.argv[6])]
        }
else:
    parameters = {
    #DEfine the number of trajectories to use
        'num_trajectories' : 0,
    
        'num_learning_epochs' : 2,
    
        'num_trials' : 2,
    
        'res' : 8,
    
        'sample' : 5,
        
        'net' : cnn_gru #Number of samples to drew
        }
    
print(parameters)
for key,val in parameters.items():
    exec(key + '=val')

num_trajectories = num_trajectories    
num_learning_epochs = num_learning_epochs
num_trials = num_trials
res = res
sample = sample
chosen_net = net


train_dataframe = pd.DataFrame()
test_dataframe = pd.DataFrame()
train_prll_dataframe = pd.DataFrame()
test_prll_dataframe = pd.DataFrame()
test_dataframe_no_coordinates = pd.DataFrame()
test_dataframe_no_coordinates_prll = pd.DataFrame()



for trial in range(num_trials):
    print('###################### Trial number {} with {}###########################'.format(trial,chosen_net().name))
    print("concat = True")
    net = chosen_net(n_timesteps = sample, hidden_size = 256,input_size = res,
            cnn_dropout=0.4,rnn_dropout=0.2, lr = 1e-4,
            concat = True)
    
    train_accuracy = []
    test_accuracy = []
    test_no_coor_accuracy = []
    
    print("concat = False")
    rnn_net = chosen_net(n_timesteps = sample, hidden_size = 256,input_size = res,
            cnn_dropout=0.4,rnn_dropout=0.2, lr = 1e-4,
            concat = False)
    train_accuracy_prll = []
    test_accuracy_prll = []
    test_no_coor_accuracy_prll = []
        
    train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                                       sample = sample,
                                                       return_datasets=True, 
                                                       add_seed = 0, 
                                                       mixed_state = False)
    train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample)
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset, sample)
    print("##########Fit {} and trajectories model on training data######".format(net.name))
    history = net.fit(
        train_dataset_x,
        train_dataset_y,
        batch_size=64,
        epochs=num_learning_epochs,
        # We pass some validation for128
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(test_dataset_x, test_dataset_y),
        verbose = 0) #(validation_images, validation_labels)
    print('Validation Accuracy = ',history.history['val_sparse_categorical_accuracy'])
    train_accuracy.append(history.history['sparse_categorical_accuracy'])
    test_accuracy.append(history.history['val_sparse_categorical_accuracy'])
    results = net.evaluate((test_dataset_x[0],test_dataset_x[1]*0), test_dataset_y, verbose = 0)
    print('When zeroing out the trajectories test accuracy = ', results[1])
    test_no_coor_accuracy.append(results[1])
    
    print("##################Fit {} model on training data##############".format(rnn_net.name))
    history2 = rnn_net.fit(
        train_dataset_x,
        train_dataset_y,
        batch_size=64,
        epochs=num_learning_epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(test_dataset_x, test_dataset_y),
        verbose = 0) #(validation_images, validation_labels)
    
    print('Validation Accuracy = ',history2.history['val_sparse_categorical_accuracy'])
    
    results2 = rnn_net.evaluate((test_dataset_x[0],test_dataset_x[1]*0), test_dataset_y, verbose = 0)
    train_accuracy_prll.append(history2.history['sparse_categorical_accuracy'])
    test_accuracy_prll.append(history2.history['val_sparse_categorical_accuracy'])
    print('When zeroing out the trajectories test accuracy = ', results2[1])
    test_no_coor_accuracy_prll.append(results2[1])
    train_dataframe['trial_{}'.format(trial)] = train_accuracy[0]
    test_dataframe['trial_{}'.format(trial)] = test_accuracy[0]
    train_prll_dataframe['trial_{}'.format(trial)]  = train_accuracy_prll[0]
    test_prll_dataframe['trial_{}'.format(trial)]  = test_accuracy_prll[0]
    test_dataframe_no_coordinates['trial_{}_train_loss'.format(trial)] = test_no_coor_accuracy[0]
    test_dataframe_no_coordinates_prll['trial_{}_test_accur'.format(trial)] = test_accuracy_prll[0]

#########################   Save Data   ######################################
train_dataframe.to_pickle('concat_True_train_dataset_{}_{}'.format(num_trajectories,sample))
test_dataframe.to_pickle('concat__True_test_dataset_{}_{}'.format(num_trajectories,sample))
train_prll_dataframe.to_pickle('concat_False_train_prll_dataframe{}_{}'.format(num_trajectories,sample))
test_prll_dataframe.to_pickle('test_False_prll_dataframe{}_{}'.format(num_trajectories,sample))
test_dataframe_no_coordinates.to_pickle('concat_True_test_dataset_no_coordinates{}_{}'.format(num_trajectories,sample))
test_dataframe_no_coordinates_prll.to_pickle('concat_False_test_dataframe_no_coordinates_prll{}_{}'.format(num_trajectories,sample))
#########################   Plot Data    ######################################

###########################Plot train##########################################
train_dataframe['mean'] = train_dataframe.mean(numeric_only = True, axis = 1)
train_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(train_dataframe) - 1, loc = train_dataframe.mean(axis = 1), scale = st.sem(train_dataframe, axis = 1))[0]
train_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(train_dataframe) - 1, loc = train_dataframe.mean(axis = 1), scale = st.sem(train_dataframe, axis = 1))[1]
plt.figure()
x = np.arange(len(train_dataframe))
y = train_dataframe['mean']
plt.plot(x, y,label = '{}'.format(net.name))
plt.fill_between(x, train_dataframe['confidance-'] , train_dataframe['confidance+'], alpha = 0.4)

train_prll_dataframe['mean'] = train_prll_dataframe.mean(numeric_only = True, axis = 1)
train_prll_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(train_prll_dataframe) - 1, loc = train_prll_dataframe.mean(axis = 1), scale = st.sem(train_prll_dataframe, axis = 1))[0]
train_prll_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(train_prll_dataframe) - 1, loc = train_prll_dataframe.mean(axis = 1), scale = st.sem(train_prll_dataframe, axis = 1))[1]
x = np.arange(len(train_prll_dataframe))
y = train_prll_dataframe['mean']
plt.plot(x, y,label = '{}'.format(rnn_net.name))
plt.fill_between(x, train_prll_dataframe['confidance-'] , train_prll_dataframe['confidance+'], alpha = 0.4)
plt.title('Mean train accuracy from {} trials \n with {} concat = True/False'.format(num_trials, num_trajectories))
plt.legend()
plt.savefig('cifar_concat_{}_train_accuracy.png'.format(net.name))

############################Plot Test##########################################
test_dataframe['mean'] = test_dataframe.mean(numeric_only = True, axis = 1)
test_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_dataframe) - 1, loc = test_dataframe.mean(axis = 1), scale = st.sem(test_dataframe, axis = 1))[0]
test_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_dataframe) - 1, loc = test_dataframe.mean(axis = 1), scale = st.sem(test_dataframe, axis = 1))[1]
plt.figure()
x = np.arange(len(test_dataframe))
y = test_dataframe['mean']
plt.plot(x, y,label = 'True')
plt.fill_between(x, test_dataframe['confidance-'] , test_dataframe['confidance+'],alpha = 0.4)

test_prll_dataframe['mean'] = test_prll_dataframe.mean(numeric_only = True, axis = 1)
test_prll_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_prll_dataframe) - 1, loc = test_prll_dataframe.mean(axis = 1), scale = st.sem(test_prll_dataframe, axis = 1))[0]
test_prll_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_prll_dataframe) - 1, loc = test_prll_dataframe.mean(axis = 1), scale = st.sem(test_prll_dataframe, axis = 1))[1]
x = np.arange(len(test_prll_dataframe))
y = test_prll_dataframe['mean']
plt.plot(x, y,label = 'False')
plt.fill_between(x, test_prll_dataframe['confidance-'] , test_prll_dataframe['confidance+'], alpha = 0.4)

# test_dataframe_no_coordinates['mean'] = test_dataframe_no_coordinates.mean(numeric_only = True, axis = 1)
# test_dataframe_no_coordinates['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_dataframe_no_coordinates) - 1, loc = test_dataframe_no_coordinates.mean(axis = 1), scale = st.sem(test_dataframe_no_coordinates, axis = 1))[0]
# test_dataframe_no_coordinates['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_dataframe_no_coordinates) - 1, loc = test_dataframe_no_coordinates.mean(axis = 1), scale = st.sem(test_dataframe_no_coordinates, axis = 1))[1]
# x = np.arange(len(test_dataframe_no_coordinates))
# y = test_dataframe_no_coordinates['mean']
# plt.plot(x, y,label = 'Regular RNN no coordinates integration')
# plt.fill_between(x, test_dataframe_no_coordinates['confidance-'] , test_dataframe_no_coordinates['confidance+'],alpha = 0.4)

# test_dataframe_no_coordinates_prll['mean'] = test_dataframe_no_coordinates_prll.mean(numeric_only = True, axis = 1)
# test_dataframe_no_coordinates_prll['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_dataframe_no_coordinates_prll) - 1, loc = test_dataframe_no_coordinates_prll.mean(axis = 1), scale = st.sem(test_dataframe_no_coordinates_prll, axis = 1))[0]
# test_dataframe_no_coordinates_prll['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_dataframe_no_coordinates_prll) - 1, loc = test_dataframe_no_coordinates_prll.mean(axis = 1), scale = st.sem(test_dataframe_no_coordinates_prll, axis = 1))[1]
# x = np.arange(len(test_dataframe_no_coordinates_prll))
# y = test_dataframe_no_coordinates_prll['mean']
# plt.plot(x, y,label = 'Regular RNN no coordinates integration')
# plt.fill_between(x, test_dataframe_no_coordinates_prll['confidance-'] , test_dataframe_no_coordinates_prll['confidance+'],alpha = 0.4)

plt.title('Mean test accuracy from {} trials \n with {} Concat = True/False'.format(num_trials, num_trajectories))
plt.legend()
plt.savefig('cifar_concat_{}_test_accuracy.png'.format(net.name))

#########################   Save Data   ######################################
train_dataframe.to_pickle('concat_True_train_dataset_{}_{}'.format(num_trajectories,sample))
test_dataframe.to_pickle('concat__True_test_dataset_{}_{}'.format(num_trajectories,sample))
train_prll_dataframe.to_pickle('concat_False_train_prll_dataframe{}_{}'.format(num_trajectories,sample))
test_prll_dataframe.to_pickle('test_False_prll_dataframe{}_{}'.format(num_trajectories,sample))
test_dataframe_no_coordinates.to_pickle('concat_True_test_dataset_no_coordinates{}_{}'.format(num_trajectories,sample))
test_dataframe_no_coordinates_prll.to_pickle('concat_False_test_dataframe_no_coordinates_prll{}_{}'.format(num_trajectories,sample))
