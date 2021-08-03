#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:12:43 2021

@author: orram
"""
import os 
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')

sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

import pickle

print(tf.__version__)
# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
images, labels = trainX, trainY
# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)

#path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/'
path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'
def net():
    input = keras.layers.Input(shape=(32,32,3))

    #Define CNN
    x = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn1')(input)
    x = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2), 
                            name = 'max_pool1')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn2')(x)
    x = keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn22')(x)
    x = keras.layers.MaxPooling2D((2, 2), 
                            name = 'max_pool2')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn3')(x)
    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn32')(x)
    x = keras.layers.MaxPooling2D((2, 2), 
                            name = 'max_pool3')(x)
    x = keras.layers.Dropout(0.2)(x)
    #Flatten and add linear layer and softmax'''



    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128,activation="relu", 
                            name = 'fc1')(x)
    x = keras.layers.Dense(10,activation="softmax", 
                            name = 'final')(x)

    model = keras.models.Model(inputs=input,outputs=x)
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

teacher = keras.models.load_model(path + 'cifar_trained_model')
teacher.evaluate(trainX[45000:], trainY[45000:], verbose=2)

def save_model(net,path,parameters,checkpoint = True):
    feature = parameters['feature']
    traject = parameters['trajectory_index']
    home_folder = path + '{}_{}_saved_models/'.format(feature, traject)
    os.mkdir(home_folder)
    if checkpoint:
        child_folder = home_folder + 'checkpoint/'
    else:
        child_folder = home_folder + 'end_of_run_model/'
    os.mkdir(child_folder)
    
    #Saving using net.save method
    model_save_path = child_folder + '{}_keras_save'.format(feature)
    os.mkdir(model_save_path)
    net.save(model_save_path)
    #LOADING WITH - keras.models.load_model(path)
    
    #Saving weights as numpy array
    numpy_weights_path = child_folder + '{}_numpy_weights/'.format(feature)
    os.mkdir(numpy_weights_path)
    all_weights = net.get_weights()
    with open(numpy_weights_path + 'numpy_weights_{}_{}'.format(feature,traject), 'wb') as file_pi:
        pickle.dump(all_weights, file_pi)
    #LOAD WITH - pickle.load - and load manualy to model.get_layer.set_weights()
    
    #save weights with keras
    keras_weights_path = child_folder + '{}_keras_weights/'.format(feature)
    os.mkdir(keras_weights_path)
    net.save_weights(keras_weights_path + 'keras_weights_{}_{}'.format(feature,traject))
    #LOADING WITH - load_status = sequential_model.load_weights("ckpt")

if len(sys.argv) == 1:
    parameters = {
    'layer_name' : 'max_pool2',#layers_names[int(sys.argv[1])],
    'feature' : 3,#int(sys.argv[2]),
    'trajectory_index' : 40,#int(sys.argv[3]),
    'run_index' : np.random.randint(10,100),
    'dropout' : 0,
    'rnn_dropout' : 0
    }
save_model(teacher, path, parameters)
    
    
    































