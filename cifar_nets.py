#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cifar syclop selected nets
"""
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
import numpy as np

import tensorflow.keras as keras
import tensorflow as tf


def cnn_gru(n_timesteps = 5, hidden_size = 128,input_size = 32,
            cnn_dropout=0.4,rnn_dropout=0.2, lr = 5e-4,
            concat = True):
    '''
    
    CNN GRU combination that extends the CNN to a network that achieves 
    ~80% accuracy on full res cifar.
    Reaches 62% on low_res syclop with hyperparameters:
        hs = 256, sample_size = 10, 
        cnn_dropout = 0.4 and rnn dropout = 0.2 
        lr = 5e-4 
        res = 8

    '''
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    print(x1.shape)


    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)

    # define LSTM model
    x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),
                         return_sequences=True,recurrent_dropout=rnn_dropout, 
                         kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'cnn_gru_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def parallel_gru(n_timesteps = 10, hidden_size = 256,input_size = 8,
            cnn_dropout=0.4,rnn_dropout=0.2, lr = 5e-4,
            concat = True):
    '''
    
    CNN RNN combination that extends the CNN to a network that achieves 
    ~80% accuracy on full res cifar.

    Parameters
    ----------
    n_timesteps : TYPE, optional
        DESCRIPTION. The default is 5.
    img_dim : TYPE, optional
        DESCRIPTION. The default is 32.
    hidden_size : TYPE, optional
        DESCRIPTION. The default is 128.
    input_size : TYPE, optional
        DESCRIPTION. The default is 32.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    '''
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    
    ###################### CNN Chanell 1#######################################
    
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3), activation='relu',padding = 'same'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3), activation='relu',padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    
    ###################### Parallel Chanell 1##################################
    rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    if concat:
        rnn_temp = keras.layers.Concatenate()([rnn_temp,inputB])
    else:
        rnn_temp = rnn_temp
    print('flat shape after cnn1', rnn_temp.shape)
    rnn_x = keras.layers.GRU( hidden_size,input_shape=(n_timesteps, None),
                             kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                             return_sequences=True,recurrent_dropout=2*rnn_dropout,
                             )(rnn_temp)
    print('gru hidden states 1 ', rnn_x.shape)
    ###################### CNN Chanell 2 #######################################
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2),name = 'test'),name = 'test')(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    
    ###################### Parallel Chanell 2 ##################################
    rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print('flat shape after cnn2',rnn_temp.shape)  
    if concat:
        rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp,inputB])
    else:
        rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp])
    print(' cnn2 input combined with fst hidden state', rnn_temp.shape)
    rnn_x = keras.layers.GRU( hidden_size,input_shape=(n_timesteps, None),
                             kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                             return_sequences=True,recurrent_dropout=2*rnn_dropout,
                             )(rnn_temp)
    print('gru hidden states 2 ', rnn_x.shape)
    
    ###################### CNN Chanell 3 #######################################
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    print(x1.shape)
    
    ###################### Parallel Chanell 3 ##################################
    # rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    # print('flat shape after cnn3',rnn_temp.shape)
    # if concat:
    #     rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp,inputB])
    # else:
    #     rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp])
    # print(' cnn23input combined with snd hidden state', rnn_temp.shape)
    # rnn_x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=2*rnn_dropout)(rnn_temp)
    # print('gru hidden states 3 ', rnn_x.shape)
    
    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)

    if concat:
        x = keras.layers.Concatenate()([x1,rnn_x,inputB])
    else:
        x = keras.layers.Concatenate()([x1,rnn_x])
    print(x.shape)

    # define LSTM model
    x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=rnn_dropout)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'parallel_gru_v1_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model