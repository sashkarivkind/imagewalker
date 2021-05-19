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

import torch
from torch.optim import Adam, SGD
import torch.nn as nn

import tensorflow as tf
import tensorflow.keras as keras

from mnist import MNIST

from utils import * 


#DEfine the number of trajectories to use
num_trajectories = int(sys.argv[1])

num_learning_epochs =  int(sys.argv[2])

num_trials = int(sys.argv[3])

res = int(sys.argv[4])
#define the place holders to hold the detials of each run 
#One dataframe for the RNN eith the coordinates insertes
#columns_train = []
#columns_test = []
#columns_test_no_ccor
#for trial in range(num_trials):
#    columns_train.append('trial_{}_train_loss'.format(trial))
#    columns_test.append('trial_{}_test_accur'.format(trial))
#    columns_test_no_ccor.append('trial_{}_no_coord_test_accur'.format(trial))
    
    
train_dataset = pd.DataFrame()
test_dataset = pd.DataFrame()
test_no_coor_dataset = pd.DataFrame()
#The second dataframe is for the RNN without the coordinates
columns = []

for trial in range(num_trials):
    columns.append('trial_{}_train_loss'.format(trial))
    columns.append('trial_{}_test_accur'.format(trial))
    
train_dataset_no_coordinates = pd.DataFrame()
test_dataset_no_coordinates = pd.DataFrame()



mnist = MNIST('/home/labs/ahissarlab/orra/datasets/mnist')
images, labels = mnist.load_training()

def train(net, epochs):
    lr = 3e-3
    #net = CNN().double()
    optimizer = Adam(net.parameters(), lr = lr)
    loss_func = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
    #Create a list to hold the q_seq example, the q_seq always holds the last q_seq 
    #of the dataframe that we created, if it is the same not good.
    q_seq_list = []
    train_loss = []
    no_traject_test_accuracy = []
    test_accur = []
    for epoch in range(epochs):
        train_dataloader, test_dataloader, ts_train, train_labels, q_sequence = \
            create_mnist_dataset(images, labels, res = res, add_seed = num_trajectories)
        q_seq_list.append(q_sequence)
        batch_loss = []
        for batch_idx, (data, traject, targets) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                data = data.to('cuda', non_blocking=True)
                targets = targets.to('cuda', non_blocking = True)
                traject = traject.to('cuda', non_blocking = True)
            #print(batch_idx, data.shape, targets.shape)
            if net.__class__.__name__ == 'RNN_Net':
                data = data.unsqueeze(2)
            optimizer.zero_grad()
            output = net(data.double(), traject.double())
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            
    
        train_loss.append(np.mean(batch_loss))

        if epoch%1 == 0:
            correct = 0
            no_traject_correct = 0
            test_accuracy = []
            for batch_idx, (test_data, test_traject, test_targets) in enumerate(test_dataloader):
                if torch.cuda.is_available():
                    test_data = test_data.to('cuda', non_blocking=True)
                    test_targets = test_targets.to('cuda', non_blocking = True)
                    test_traject = test_traject.to('cuda', non_blocking = True)
                #print(batch_idx, data.shape, targets.shape)
                if net.__class__.__name__ == 'RNN_Net':
                    test_data = test_data.unsqueeze(2)
                #Run Regular Test##############################################
                test_output = net(test_data,test_traject)
                test_pred = test_output.data.max(1, keepdim = True)[1]
                correct = test_pred.eq(test_targets.data.view_as(test_pred)).sum()
                test_accuracy.append(100.*correct.to('cpu')/len(test_targets))
                #Run Test without Trajectories ###############################
                no_traject_test_output = net(test_data,test_traject*0)
                no_traject_test_pred = no_traject_test_output.data.max(1, keepdim = True)[1]
                no_traject_correct = no_traject_test_pred.eq(test_targets.data.view_as(no_traject_test_pred)).sum()
                no_traject_test_accuracy.append(100.*no_traject_correct.to('cpu')/len(test_targets))

            print('Net',net.__class__.__name__,'Epoch : ',epoch+1, '\t', 'loss :', loss.to('cpu').item(), 'accuracy :',np.mean(test_accuracy) )
            test_accur.append(np.mean(test_accuracy))
        
    return train_loss, test_accur , no_traject_test_accuracy, q_seq_list

for trial in range(num_trials):
    print("RNN + Trajectories")
    net = RNN_Net(traject = True).double()
    train_loss, test_accur, no_traject_test_accuracy,q_seq_list = \
        train(net, epochs = num_learning_epochs)
    train_dataset['trial_{}_train_loss'.format(trial)] = train_loss
    test_dataset['trial_{}_test_accur'.format(trial)] = test_accur
    test_no_coor_dataset['trial_{}_no_coord_test_accur'.format(trial)]  = no_traject_test_accuracy
    
    print("Only RNN")
    net_no = RNN_Net(traject = False).double()
    train_loss, test_accur, _ , _= \
        train(net_no, epochs = num_learning_epochs)
    train_dataset_no_coordinates['trial_{}_train_loss'.format(trial)] = train_loss
    test_dataset_no_coordinates['trial_{}_test_accur'.format(trial)] = test_accur

    
######################### Plot and Save ######################################

#Plot
train_dataset['mean'] = train_dataset.mean(numeric_only = True, axis = 1)
train_dataset['confidance-'] = st.t.interval(alpha = 0.95, df = len(train_dataset) - 1, loc = train_dataset.mean(axis = 1), scale = st.sem(train_dataset, axis = 1))[0]
train_dataset['confidance+'] = st.t.interval(alpha = 0.95, df = len(train_dataset) - 1, loc = train_dataset.mean(axis = 1), scale = st.sem(train_dataset, axis = 1))[1]
plt.figure()
x = np.arange(len(train_dataset))
y = train_dataset['mean']
plt.plot(x, y)
plt.fill_between(x, train_dataset['confidance-'] , train_dataset['confidance+'])
plt.savefig('train_accuracy_{}.png'.format(num_trajectories))

test_dataset['mean'] = test_dataset.mean(numeric_only = True, axis = 1)
test_dataset['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_dataset) - 1, loc = test_dataset.mean(axis = 1), scale = st.sem(test_dataset, axis = 1))[0]
test_dataset['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_dataset) - 1, loc = test_dataset.mean(axis = 1), scale = st.sem(test_dataset, axis = 1))[1]
plt.figure()
x = np.arange(len(test_dataset))
y = test_dataset['mean']
plt.plot(x, y, label = 'with coordinates')
plt.fill_between(x, test_dataset['confidance-'] , test_dataset['confidance+'])

test_no_coor_dataset['mean'] = test_no_coor_dataset.mean(numeric_only = True, axis = 1)
test_no_coor_dataset['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_no_coor_dataset) - 1, loc = test_no_coor_dataset.mean(axis = 1), scale = st.sem(test_no_coor_dataset, axis = 1))[0]
test_no_coor_dataset['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_no_coor_dataset) - 1, loc = test_no_coor_dataset.mean(axis = 1), scale = st.sem(test_no_coor_dataset, axis = 1))[1]
plt.figure()
x = np.arange(len(test_no_coor_dataset))
y = test_no_coor_dataset['mean']
plt.plot(x, y, label = 'without coordinates')
plt.fill_between(x, test_no_coor_dataset['confidance-'] , test_no_coor_dataset['confidance+'])
plt.legend
plt.savefig('test_accuracy_{}.png'.format(num_trajectories))


