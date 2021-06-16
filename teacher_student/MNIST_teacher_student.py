#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teacher Student network

Train a student who sees low resolurtion images to extract features learnt from
a teacher who learns from high resolution images. 

Step 1 :
    Train CNN on HR MNIST
    Train smaller CNN on LR MNIST
    Start Teacher Learning - 
        Insert HR/LR couple one to the HR one to the LR
        Create a MSE loss from the last conv layer of the HR to the last
        Conv layer of the LR (same size) 
        Continue training combining the two loss (MSE and Classification lost)
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
import sys
import os 
import matplotlib.pyplot as plt

sys.path.insert(1, "/home/orram/Documents/GitHub/imagewalker/")#'/home/labs/ahissarlab/orra/imagewalker')


from keras_utils import *
from misc import *

import importlib
importlib.reload(misc)

import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from mnist import MNIST

# load dataset
#mnist = MNIST('/home/labs/ahissarlab/orra/datasets/mnist')#MNIST('/home/orram/Documents/datasets/MNIST')
from tensorflow.keras.datasets import mnist
(trainX, trainy), (testX, testy) = mnist.load_data()


# fmnist = torchvision.datasets.FashionMNIST('/home/orram/Documents/datasets/fmnist', train = True, download = True)
# images, labels = fmnist.data, fmnist.targets

#%%
########################### Train HD CNN #####################################
print('############## Training HD Teacher ###################################')
HD_cnn = HRcnn(input_size = 28, input_dim = 1)

HD_history = HD_cnn.fit(
    trainX,
    trainy,
    batch_size = 64,
    epochs = 4,
    validation_data = (testX,testy,),
    verbose = 1)

#%%
######################### Get Features from HD model #########################
print('############## Get Features from HD model ###################################')
layer_name = 'teacher_features'
HD_features_model = keras.Model(inputs=HD_cnn.input,
                                       outputs=HD_cnn.get_layer(layer_name).output)
HD_features = HD_features_model(trainX)

#%%
######################### Train LR model #####################################
print('############## Create LR dataset Student ###################################')
import cv2
LR_trainX = []
for img in trainX:
    #img = np.reshape(img, [1,28,28])
    img = cv2.resize(img,(6,6), interpolation = cv2.INTER_CUBIC)
    img = cv2.resize(img,(28,28), interpolation = cv2.INTER_CUBIC)
    LR_trainX.append(img)

LR_trainX = np.array(LR_trainX)
#%%

print('train the trained teacher model on new LR images:')
HD_history = HD_cnn.fit(
    LR_trainX,
    trainy,
    batch_size = 64,
    epochs = 4,
    validation_data = (testX,testy,),
    verbose = 1)

print('train a new model on the LR images')
LR_cnn = HRcnn(input_size = 28, input_dim = 1)
HD_history = LR_cnn.fit(
    LR_trainX,
    trainy,
    batch_size = 64,
    epochs = 4,
    validation_data = (testX,testy,),
    verbose = 1)

print('retrain the trained teacher model on HR images:')
HD_cnn = HRcnn(input_size = 28, input_dim = 1)

HD_history = HD_cnn.fit(
    trainX,
    trainy,
    batch_size = 64,
    epochs = 4,
    validation_data = (testX,testy,),
    verbose = 1)

#%%
############################ Train Student Network #############################
print('############## Train Student Network ###################################')
from keras_utils import *
student = StudentCNN(input_size = 28, input_dim = 1)

student_history = student.fit(
    LR_trainX,
    [HD_features, trainy],
    batch_size = 64,
    epochs = 1,
    )






