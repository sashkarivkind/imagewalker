'''



'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import os
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker/')
import numpy as np
import cv2
import misc
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from keras_utils import dataset_update, write_to_file, create_cifar_dataset, extended_cnn_one_img
from misc import *

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy

kernel_regularizer_list = [None, keras.regularizers.l1(),keras.regularizers.l2(),keras.regularizers.l1_l2()]
optimizer_list = [tf.keras.optimizers.Adam, tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop]
if len(sys.argv) > 1:
    paramaters = {
    'epochs' : int(sys.argv[1]),
    
    'sample' : int(sys.argv[2]),
    
    'res' : int(sys.argv[3]),
    
    'hidden_size' : int(sys.argv[4]),
    
    'concat' : int(sys.argv[5]),
    
    'regularizer' : keras.regularizers.l1(),#kernel_regularizer_list[int(sys.argv[6])],
    
    'optimizer' : optimizer_list[int(sys.argv[7])],
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4,
    
    'run_id' : np.random.randint(1000,9000)
    }
    
else:
    paramaters = {
    'epochs' : 1,
    
    'sample' : 5,
    
    'res' : 8,
    
    'hidden_size' : 128,
    
    'concat' : 1,
    
    'regularizer' : None,
    
    'optimizer' : optimizer_list[0],
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4,
    
    'run_id' : np.random.randint(1000,9000)
    }
   
print(paramaters)
for key,val in paramaters.items():
    exec(key + '=val')
epochs = epochs
sample = sample 
res = res 
hidden_size =hidden_size
concat = concat
regularizer = regularizer
optimizer = optimizer
cnn_dropout = cnn_dropout
rnn_dropout = rnn_dropout
lr = lr
run_id = run_id
n_timesteps = sample
def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def cnn_gru(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True, 
            optimizer = tf.keras.optimizers.Adam, ):
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
                         kernel_regularizer=regularizer)(x)
    
    x = keras.layers.Flatten()(x)
    #Add another dense layer (prior it reached 62%)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'cnn_gru_{}'.format(concat))
    opt=optimizer(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = cnn_gru(n_timesteps = sample, hidden_size = hidden_size,input_size = res, concat = concat)
cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = res, dropout = cnn_dropout)


# hp = HP()
# hp.save_path = 'saved_runs'

# hp.description = "syclop cifar net search runs"
# hp.this_run_name = 'syclop_{}'.format(rnn_net.name)
# deploy_logs()

train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )
                                    #bad_res_func = bad_res101, up_sample = True)

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)

print("##################### Fit {} and trajectories model on training data res = {} ##################".format(cnn_net.name,res))
cnn_history = cnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 0)

print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(cnn_net.name),cnn_history.history['sparse_categorical_accuracy'])


print("##################### Fit {} and trajectories model on training data res = {} ##################".format(rnn_net.name,res))
rnn_history = rnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 0)

print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
# plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
# plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
plt.legend()
plt.grid()
plt.ylim(0.5,0.7)
plt.title('{} on cifar res = {} hs = {} dropout = {}, num samples = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout,sample))
plt.savefig('{} on Cifar res = {}, no upsample, val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}'.format(run_id), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
# with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict'.format(cnn_net.name), 'wb') as file_pi:
#     pickle.dump(cnn_history.history, file_pi)


dataset_update(rnn_history, rnn_net,paramaters)    
write_to_file(rnn_history, rnn_net,paramaters)    
    
    