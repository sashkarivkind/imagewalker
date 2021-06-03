#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:38:04 2021

@author: orram
"""

import numpy as np
import pandas as pd
import sys
import os 
import matplotlib.pyplot as plt

sys.path.insert(1, "/home/orram/Documents/GitHub/imagewalker/")#'/home/labs/ahissarlab/orra/imagewalker')

from keras_utils import *
from misc import *

import tensorflow.keras as keras
import tensorflow as tf


# load dataset
#mnist = MNIST('/home/labs/ahissarlab/orra/datasets/mnist')#MNIST('/home/orram/Documents/datasets/MNIST')
from tensorflow.keras.datasets import mnist
(trainX, trainy), (testX, testy) = mnist.load_data()
trainX, testX = trainX/255.0, testX/255.0
#%%
# teacher = keras.Sequential([keras.layers.Conv2D(16,3,activation = 'relu'),
#                            keras.layers.MaxPool2D(),
#                            keras.layers.Dropout(0.2),
#                            keras.layers.Conv2D(32,3,activation = 'relu'),
#                            keras.layers.MaxPool2D(),
#                            keras.layers.Dropout(0.2),
#                            keras.layers.Conv2D(16,3, activation = 'relu', name = 'teacher_features'), 
#                            keras.layers.Flatten(),
#                            keras.layers.Dense(10, activation = 'softmax'),
#                            ],
#                            name = 'teacher'
 
#                           )
def teacher_model(input_size = 28, input_dim = 1):
    inputA = keras.layers.Input(shape=(input_size, input_size, input_dim))
    
    x1 = keras.layers.Conv2D(16, kernel_size = 3, activation = 'relu')(inputA)
    #x = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.MaxPool2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu')(x1)
    #x = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.MaxPool2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Conv2D(16, kernel_size = 3, activation = 'relu', )(x1)
    x = keras.layers.Dropout(0.2)(x1)
    
    x = keras.layers.Flatten(name = 'teacher_features')(x)
    
    x = keras.layers.Dense(10, activation = 'softmax')(x)
    
    model = keras.models.Model(inputs = inputA, outputs = x, name = 'teacher')
    
    opt = tf.keras.optimizers.Adam(lr = 3e-3)
    
    model.compile( optimizer = opt,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['sparse_categorical_accuracy'])
    
    return model

print('############## Training HD Teacher ###################################')
teacher = teacher_model(input_size = 28, input_dim = 1)

teacher_history = teacher.fit(
    trainX,
    trainy,
    batch_size = 64,
    epochs = 1,
    validation_data = (testX,testy,),
    verbose = 1)

feature_teacher = keras.models.Model(inputs = teacher.inputs, outputs = teacher.layers[-4].output)

#%%
def teacher_student():
    LR_img = keras.Input(shape = (28,28,1), name = 'LR_img')

    student_features_model = keras.Sequential([keras.layers.Conv2D(16,3,activation = 'relu'),
                           keras.layers.MaxPool2D(),
                           keras.layers.Dropout(0.2),
                           keras.layers.Conv2D(32,3,activation = 'relu'),
                           keras.layers.MaxPool2D(),
                           keras.layers.Dropout(0.2),
                           keras.layers.Conv2D(16,3, activation = 'relu'),
                           keras.layers.Flatten(),
                           ],
                                              name = 'student_features_model')
    
    student_dense = keras.Sequential([
                                     keras.layers.Dense(10, activation='softmax'),
                                     ],
                                     name = 'student_dense'
                                     )
    

    student_features = student_features_model(LR_img)
    print(student_features.shape)
    student_pred = student_dense(student_features)
    
    model = keras.Model(inputs = [LR_img],
                        outputs = [
                                student_features,
                                   student_pred,],
                        )
    
    
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss=["mean_squared_error","sparse_categorical_crossentropy"],
        metrics=["mean_squared_error","sparse_categorical_accuracy"],
    )#"mean_squared_error",
    return model

def feature_loss_func(y_true, y_pred):
    feature_loss = keras.backend.mean(keras.losses.mean_squared_error(y_true, y_pred))
    return feature_loss

ts = teacher_student()   

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
                       
print('############## Get Features from HD model ###################################')
layer_name = 'teacher_features'
HD_features_model = keras.Model(inputs=teacher.input,
                                       outputs=teacher.get_layer(layer_name).output)
HD_features = HD_features_model(trainX)
HD_features_test = HD_features_model(testX)

#%%
ts = teacher_student() 
ts_history = ts.fit(
    LR_trainX,
   [ trainy,HD_features],
    batch_size = 64,
    epochs = 1,
    validation_data = (LR_testX,[testy,HD_features_test]),
    verbose = 1)         

                  