#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:57:00 2021

@author: orram
"""

import numpy as np
import pandas as pd
import sys
import os 
import matplotlib.pyplot as plt

sys.path.insert(1, "/home/orram/Documents/GitHub/imagewalker/")#'/home/labs/ahissarlab/orra/imagewalker')

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets



mnist_data = datasets.MNIST('/home/orram/',transform = torchvision.transforms.ToTensor())

HR_data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=64,
                                          shuffle=True,
                                           )

mnist_test = datasets.MNIST('/home/orram/', train = False,transform = torchvision.transforms.ToTensor())

HR_test_loader = torch.utils.data.DataLoader(mnist_test,
                                          batch_size=10_000,
                                          shuffle=True,
                                         
                                          )

#%%
epochs = 1


class teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,16,3,stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(2)
        
        #After the layers and pooling the first two we should get 
        # 16,3,3
        #Flatting it we get:
        # 144
        
        self.fc1 = nn.Linear(8*8*16,64)
        self.fc2 = nn.Linear(64,10)
        
        self.relu = nn.ReLU()
        
    def forward(self, img):
        
        img = self.pool(self.relu(self.bn1(self.conv1(img.double()))))
        img = self.pool(self.relu(self.bn2(self.conv2(img))))
        img = self.relu(self.bn3(self.conv3(img)))        
        #print(img.shape)
        img_features = img.view(img.shape[0],8*8*16)
        img = self.relu(self.fc1(img_features))
        img = self.fc2(img)
        
        return img,img_features
    
class student(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,16,3,stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(2)
        
        #After the layers and pooling the first two we should get 
        # 16,3,3
        #Flatting it we get:
        # 144
        
        self.fc1 = nn.Linear(8*8*16,64)
        self.fc2 = nn.Linear(64,10)
        
        self.relu = nn.ReLU()
        
    def forward(self, img):
        
        img = self.pool(self.relu(self.bn1(self.conv1(img.double()))))
        img = self.pool(self.relu(self.bn2(self.conv2(img))))
        img = self.relu(self.bn3(self.conv3(img)))        
        #print(img.shape)
        img_features = img.view(img.shape[0],8*8*16)
        img = self.relu(self.fc1(img_features))
        img = self.fc2(img)
        
        return img,img_features    


#%%
print('############## Training HD Teacher ###################################')
lr = 3e-3
net = teacher().double()
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()
train_loss = []
train_accur = []
test_loss = []
test_accur = []
for epoch in range(epochs):
    batch_loss = []
    train_accuracy = []
    correct = 0
    for batch_idx, (data,targets) in enumerate(HR_data_loader):
        optimizer.zero_grad()
        output = net(data.double())
        features = output[1]
        output = output[0]
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        pred = output.data.max(1, keepdim = True)[1]
        correct = pred.eq(targets.data.view_as(pred)).sum()
        train_accuracy.append(100.*correct.to('cpu')/len(targets))
    train_loss.append(np.mean(batch_loss))
    train_accur.append(np.mean(train_accuracy))
    correct = 0
    test_batch_loss = []
    test_accuracy = []
    for batch_idx, (test_data,test_targets) in enumerate(HR_test_loader):
        test_output = net(test_data)
        test_features = test_output[1]
        test_output = test_output[0]
        loss = loss_func(test_output, test_targets)
        test_batch_loss.append(loss.item())
        test_pred = test_output.data.max(1, keepdim = True)[1]
        correct = test_pred.eq(test_targets.data.view_as(test_pred)).sum()
        test_accuracy.append(100.*correct.to('cpu')/len(test_targets))

    print('Net',net.__class__.__name__,'Epoch : ',epoch+1, '\t', 'loss :', loss.to('cpu').item(), 'accuracy :',np.mean(test_accuracy) )
    test_loss.append(np.mean(test_batch_loss))
    test_accur.append(np.mean(test_accuracy))

#%%
print('############## Create LR dataset Student ###################################')
import cv2
LR_trainX = []
for img in trainX:
    #img = np.reshape(img, [1,28,28])
    img = cv2.resize(img,(3,3), interpolation = cv2.INTER_CUBIC)
    img = cv2.resize(img,(28,28), interpolation = cv2.INTER_CUBIC)
    LR_trainX.append(img)

LR_trainX = np.array(LR_trainX)

LR_testX = []
for img in testX:
    #img = np.reshape(img, [1,28,28])
    img = cv2.resize(img,(3,3), interpolation = cv2.INTER_CUBIC)
    img = cv2.resize(img,(28,28), interpolation = cv2.INTER_CUBIC)
    LR_testX.append(img)

LR_testX = np.array(LR_testX)