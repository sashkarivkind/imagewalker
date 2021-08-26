#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 20:51:15 2021

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
from vanvalenlab_convolutional_recurrent import ConvGRU2D


def net_weights_reinitializer(model):
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and \
                hasattr(model.layers[ix], 'bias_initializer'):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer

            old_weights, old_biases = model.layers[ix].get_weights()
            ev=keras.backend.eval #evaluate everything to comply with keras1
            model.layers[ix].set_weights([
                ev(weight_initializer(ev(old_weights.shape))),
                ev(bias_initializer(ev(old_biases.shape)))])

def write_to_file(history, net,paramaters):
    file_name = 'summary_file_feature_learning.txt'
    if os.path.isfile('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name)):
        file = open('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name), 'a')
    else:
        file = open('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name), 'x')
        
    
    from datetime import datetime
    file.write('#####################\n')
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")
    file.write(net.name + '\n')
    file.write(now + '\n')
    file.write(str(paramaters) + "\n")
    min_test_error = min(history.history['val_mean_squared_error'])
    min_train_error = min(history.history['mean_squared_error'])
    summary_of_run = "min_test_error = {}, min_train_error = {} \n".\
                        format(min_test_error,min_train_error)
        
    file.write(summary_of_run)
    file.close()

def dataset_update(history, net, parameters, name = '_'):
    file_name = 'summary_dataframe_feature_learning_{}.pkl'.format(name)
    if os.path.isfile('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name)):
        dataframe = pd.read_pickle('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name))
    else:
        dataframe = pd.DataFrame()
    
    values_to_add = parameters
    values_to_add['net_name'] = net.name
    values_to_add['min_test_error'] = min(history.history['val_mean_squared_error'])
    values_to_add['min_train_error'] = min(history.history['mean_squared_error'])
    values_to_add['train_error'] = [history.history['mean_squared_error']]
    values_to_add['test_error'] = [history.history['val_mean_squared_error']]
    dataframe = dataframe.append(values_to_add, ignore_index = True)

    dataframe.to_pickle('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name))

def full_learning_dataset_update(student_history, 
                                 decoder_history,
                                 full_history,
                                 net, parameters, name = '_'):
    file_name = 'summary_dataframe_feature_learning_{}.pkl'.format(name)
    if os.path.isfile('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name)):
        dataframe = pd.read_pickle('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name))
    else:
        dataframe = pd.DataFrame()
    values_to_add = parameters
    values_to_add['net_name'] = net.name
    values_to_add['student_min_test_error'] = min(student_history.history['val_mean_squared_error'])
    values_to_add['student_min_train_error'] = min(student_history.history['mean_squared_error'])
    values_to_add['student_train_error'] = [student_history.history['mean_squared_error']]
    values_to_add['student_test_error'] = [student_history.history['val_mean_squared_error']]
    values_to_add['decoder_max_test_error'] = max(decoder_history.history['val_sparse_categorical_accuracy'])
    values_to_add['decoder_max_train_error'] = max(decoder_history.history['sparse_categorical_accuracy'])
    values_to_add['decoder_train_error'] = [decoder_history.history['sparse_categorical_accuracy']]
    values_to_add['decoder_test_error'] = [decoder_history.history['val_sparse_categorical_accuracy']]
    values_to_add['full_max_test_error'] = max(full_history.history['val_sparse_categorical_accuracy'])
    values_to_add['full_max_train_error'] = max(full_history.history['sparse_categorical_accuracy'])
    values_to_add['full_train_error'] = [full_history.history['sparse_categorical_accuracy']]
    values_to_add['full_test_error'] = [full_history.history['val_sparse_categorical_accuracy']]
    dataframe = dataframe.append(values_to_add, ignore_index = True)

    dataframe.to_pickle('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name))

def traject_learning_dataset_update(train_accur,
                                    test_accur, 
                                 decoder_history,
                                 #full_history,
                                 net, parameters, name = '_'):
    '''
    TODO combine all write to dataframe functions! 
    
    '''
    file_name = 'summary_dataframe_feature_learning_{}.pkl'.format(name)
    if os.path.isfile('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name)):
        dataframe = pd.read_pickle('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name))
    else:
        dataframe = pd.DataFrame()
    values_to_add = parameters
    values_to_add['net_name'] = net.name
    values_to_add['student_min_test_error'] = min(test_accur)
    values_to_add['student_min_train_error'] = min(train_accur)
    values_to_add['student_train_error'] = train_accur
    values_to_add['student_test_error'] = test_accur
    values_to_add['decoder_max_test_error'] = max(decoder_history.history['val_sparse_categorical_accuracy'])
    values_to_add['decoder_max_train_error'] = max(decoder_history.history['sparse_categorical_accuracy'])
    values_to_add['decoder_train_error'] = [decoder_history.history['sparse_categorical_accuracy']]
    values_to_add['decoder_test_error'] = [decoder_history.history['val_sparse_categorical_accuracy']]
    # values_to_add['full_max_test_error'] = max(full_history.history['val_sparse_categorical_accuracy'])
    # values_to_add['full_max_train_error'] = max(full_history.history['sparse_categorical_accuracy'])
    # values_to_add['full_train_error'] = [full_history.history['sparse_categorical_accuracy']]
    # values_to_add['full_test_error'] = [full_history.history['val_sparse_categorical_accuracy']]
    dataframe = dataframe.append(values_to_add, ignore_index = True)

    dataframe.to_pickle('/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_learning/{}'.format(file_name))

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
    


def student3(sample = 10, res = 8, activation = 'tanh', dropout = None, rnn_dropout = None,
             num_feature = 1, layer_norm = False , n_layers=3, conv_rnn_type='lstm',block_size = 1,
             add_coordinates = False, time_pool = False):
    #TO DO add option for different block sizes in every convcnn
    #TO DO add skip connections in the block 
    if time_pool == '0':
        time_pool = 0 
    inputA = keras.layers.Input(shape=(sample, res,res,3))
    if add_coordinates:
        inputB = keras.layers.Input(shape=(sample,res,res,2))
    else:
        inputB = keras.layers.Input(shape=(sample,2))
    if conv_rnn_type == 'lstm':
        Our_RNN_cell = keras.layers.ConvLSTM2D
    elif  conv_rnn_type == 'gru':
        Our_RNN_cell = ConvGRU2D
    else:
        error("not supported type of conv rnn cell")

    #Broadcast the coordinates to a [res,res,2] matrix and concat to x
    if add_coordinates:
        x = keras.layers.Concatenate()([inputA,inputB])
    else: 
        x = inputA
    print(x.shape)
    for ind in range(block_size):
        x = Our_RNN_cell(32,(3,3), padding = 'same', return_sequences=True,
                                dropout = dropout,recurrent_dropout=rnn_dropout, 
                            name = 'convLSTM1{}'.format(ind))(x)
    for ind in range(block_size):
        x = Our_RNN_cell(64,(3,3), padding = 'same', return_sequences=True,
                            name = 'convLSTM2{}'.format(ind),
                            dropout = dropout,recurrent_dropout=rnn_dropout,)(x)
    for ind in range(block_size):
        if ind == block_size - 1:
            if time_pool:
                return_seq = True
            else:
                return_seq = False
        else:
            return_seq = True
        x = Our_RNN_cell(num_feature,(3,3), padding = 'same', return_sequences=return_seq,
                            name = 'convLSTM3{}'.format(ind), activation=activation,
                            dropout = dropout,recurrent_dropout=rnn_dropout,)(x)
    print(return_seq)
    if time_pool:
        print(time_pool)
        if time_pool == 'max_pool':
            x = tf.keras.layers.MaxPooling3D(pool_size=(sample, 1, 1))(x)
        elif time_pool == 'average_pool':
            x = tf.keras.layers.AveragePooling3D(pool_size=(sample, 1, 1))(x)
    if layer_norm:
        x = keras.layers.LayerNormalization(axis=3)(x)

    print(x.shape)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'student_3')
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    return model




def student3_one_image(sample=10, activation = 'relu', dropout = None, num_feature = 1):
    input = keras.layers.Input(shape=(sample,8,8,3))
    choose = np.random.randint(0,sample)
    #Define CNN
    #x = keras.layers.Conv2D(1,(3,3),activation='relu', padding = 'same', 
    #                        name = 'convLSTM1')(input)
    x = keras.layers.Conv2D(32,(3,3), padding = 'same', 
                            activation=activation,name = 'conv1')(input[:,choose,:,:,:])
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv2D(64,(3,3), padding = 'same', 
                               activation=activation, name = 'conv2')(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv2D(num_feature,(3,3), padding = 'same', 
                             activation=activation, name = 'conv3',)(x)
    x = keras.layers.Dropout(dropout)(x)
    print(x.shape)
    model = keras.models.Model(inputs=input,outputs=x, name = 'student_3')
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    return model



def student32(sample = 10):
    input = keras.layers.Input(shape=(sample, 8,8,3))
    
    #Define CNN
    #x = keras.layers.Conv2D(1,(3,3),activation='relu', padding = 'same', 
    #                        name = 'convLSTM1')(input)
    x = keras.layers.ConvLSTM2D(32,(3,3), padding = 'same', return_sequences=True,
                            name = 'convLSTM1')(input)
    x = keras.layers.ConvLSTM2D(64,(3,3), padding = 'same', return_sequences=True,
                            name = 'convLSTM2')(x)
    x = keras.layers.ConvLSTM2D(1,(3,3), padding = 'same', 
                            name = 'convLSTM3')(x)
    print(x.shape)
    model = keras.models.Model(inputs=input,outputs=x, name = 'student_3')

    return model

