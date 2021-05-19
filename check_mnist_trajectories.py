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
import sys 

import torch
from torch.optim import Adam, SGD
import torch.nn as nn

import tensorflow as tf
import tensorflow.keras as keras

from utils import * 


#DEfine the number of trajectories to use
num_trajectories = 4#int(sys.argv[1])

num_learning_epochs = 4#  int(sys.argv[2])

num_trials = 2# int(sys.argv[3])

res = 6# int(sys.argv[4])
#define the place holders to hold the detials of each run
columns = ['trial_{}'.format(trial) for trial in range(num_trials)]

for trial in range(num_trials):
    columns.append('trial_{}_train_accur'.format(trial))
    columns.append('trial_{}_test_accur'.format(trial))
    columns.append('trial_{}_no_coord_test_accur'.format(trial))
    


detials_dataset = pd.DataFrame(columns = columns)


mnist = MNIST('/home/orram/Documents/datasets/MNIST/')
images, labels = mnist.load_training()

# Training Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 128

validation_index=-5000

# Network Parameters
size=None
padding_size=(128,128)
# num_input = padding_size[0]*padding_size[1] # MNIST data input (img shape: 28*28)
num_classes = None 
# dropout = 0.25 # Dropout, probability to drop a unit

def train(train_dataloader, test_dataloader, net, epochs = 10):
    lr = 3e-3
    #net = CNN().double()
    optimizer = Adam(net.parameters(), lr = lr)
    loss_func = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
    
    train_loss = []
    test_loss = []
    test_accur = []
    for epoch in range(epochs):

        batch_loss = []
        for batch_idx, (data,targets) in enumerate(train_dataloader):
            if net.__class__.__name__ == 'RNN_Net':
                data = data.unsqueeze(2)
            if torch.cuda.is_available():
                data = data.to('cuda', non_blocking=True)
                targets = targets.to('cuda', non_blocking = True)
            #print(batch_idx, data.shape, targets.shape)

            optimizer.zero_grad()
            output = net(data.double())
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())


        train_loss.append(np.mean(batch_loss))

        if epoch%1 == 0:
            correct = 0
            test_batch_loss = []
            test_accuracy = []
            for batch_idx, (test_data,test_targets) in enumerate(test_dataloader):
                if net.__class__.__name__ == 'RNN_Net':
                    test_data = test_data.unsqueeze(2)
                if torch.cuda.is_available():
                    test_data = test_data.to('cuda', non_blocking=True)
                    test_targets = test_targets.to('cuda', non_blocking = True)
                #print(batch_idx, data.shape, targets.shape)

                test_output = net(test_data)
                loss = loss_func(test_output, test_targets)
                test_batch_loss.append(loss.item())
                test_pred = test_output.data.max(1, keepdim = True)[1]
                correct = test_pred.eq(test_targets.data.view_as(test_pred)).sum()
                test_accuracy.append(100.*correct.to('cpu')/len(test_targets))

            print('Net',net.__class__.__name__,'Epoch : ',epoch+1, '\t', 'loss :', loss.to('cpu').item(), 'accuracy :',np.mean(test_accuracy) )
            test_loss.append(np.mean(test_batch_loss))
            test_accur.append(np.mean(test_accuracy))
    
    return train_loss, test_loss, test_accur

for trial in range(num_trials):
    net = RNN_Net()
    train_dataloader, test_dataloader, ts_train, train_labels, q_sequence = \
        create_mnist_dataset(images, labels, res = res)
    train_loss, test_loss, test_accur = \
    train(train_dataloader, test_dataloader,net, epochs = num_learning_epochs)
    
