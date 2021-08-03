'''
These network archtecture performed perfectly while training with a teacher
Let's see ho it performes without one.

'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')

import numpy as np
import cv2
import misc
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from keras_utils import dataset_update, write_to_file, create_cifar_dataset


import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy


#Define function for low resolution lens on syclop
def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp

def bad_res102(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_AREA)
    return dwnsmp

import importlib
importlib.reload(misc)
from misc import Logger
import os 


def deploy_logs():
    if not os.path.exists(hp.save_path):
        os.makedirs(hp.save_path)

    dir_success = False
    for sfx in range(1):  # todo legacy
        candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
        if not os.path.exists(candidate_path):
            hp.this_run_path = candidate_path
            os.makedirs(hp.this_run_path)
            dir_success = Truecnn_net = cnn_one_img(n_timesteps = sample, input_size = 28, input_dim = 1)
            break
    if not dir_success:
        error('run name already exists!')

    sys.stdout = Logger(hp.this_run_path+'log.log')
    print('results are in:', hp.this_run_path)
    print('description: ', hp.description)
    #print('hyper-parameters (partial):', hp.dict)

kernel_regularizer_list = [None, keras.regularizers.l1(),keras.regularizers.l2(),keras.regularizers.l1_l2()]
optimizer_list = [tf.keras.optimizers.Adam, tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop]
if len(sys.argv) > 1:
    paramaters = {
    'epochs' : int(sys.argv[1]),
    
    'sample' : int(sys.argv[2]),
    
    'res' : int(sys.argv[3]),
    
    'hidden_size' : int(sys.argv[4]),
    
    'concat' : int(sys.argv[5]),
    
    'regularizer' : kernel_regularizer_list[int(sys.argv[6])],
    
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

def convgru_cnn(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 3, concat = False,
            optimizer = tf.keras.optimizers.Adam):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    num_feature = 64
    # define LSTM model
    x = keras.layers.ConvLSTM2D(32,(3,3), padding = 'same', return_sequences=True,
                                dropout = cnn_dropout,recurrent_dropout=rnn_dropout, 
                            name = 'convLSTM1')(inputA)
    x = keras.layers.ConvLSTM2D(64,(3,3), padding = 'same', return_sequences=True,
                            name = 'convLSTM2',
                            dropout = cnn_dropout,recurrent_dropout=rnn_dropout,)(x)
    x = keras.layers.ConvLSTM2D(num_feature,(3,3), padding = 'same', 
                            name = 'convLSTM3', activation='relu',
                            dropout = cnn_dropout,recurrent_dropout=rnn_dropout,)(x)
    print(x.shape)
    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn3')(x)
    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn32')(x)
    x = keras.layers.MaxPooling2D((2, 2), 
                            name = 'max_pool3')(x)
    x = keras.layers.Dropout(cnn_dropout)(x)
    #Flatten and add linear layer and softmax
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128,activation="relu", 
                            name = 'fc1')(x)
    x = keras.layers.Dense(10,activation="softmax", 
                            name = 'final')(x)
    
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'convgru_cnn_{}'.format(concat))
    opt=optimizer(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = convgru_cnn(n_timesteps = sample, cell_size = hidden_size,input_size = res  , concat = concat)

#%%
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )#bad_res_func = bad_res101, up_sample = True)

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)

#%%
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
    verbose = 2)


print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
plt.legend()
plt.title('{} on cifar res = {} hs = {} dropout = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout))
plt.savefig('{} on Cifar res = {} val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}_{}'.format(rnn_net.name, run_id), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
dataset_update(rnn_history, rnn_net,paramaters)    
write_to_file(rnn_history, rnn_net,paramaters)    