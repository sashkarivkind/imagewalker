from __future__ import division, print_function, absolute_import

import numpy as np
import misc
import pandas as pd
import sys
import matplotlib.pyplot as plt



from keras_utils import *
from misc import *

import importlib
importlib.reload(misc)

import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from mnist import MNIST
# load dataset
mnist = MNIST('/home/labs/ahissarlab/orra/datasets/mnist')#MNIST('/home/orram/Documents/datasets/MNIST')
images, labels = mnist.load_training()

# fmnist = torchvision.datasets.FashionMNIST('/home/orram/Documents/datasets/fmnist', train = True, download = True)
# images, labels = fmnist.data, fmnist.targets


import importlib
importlib.reload(misc)
from misc import Logger
import os 

hp = HP()
hp.save_path = 'saved_runs'

hp.description = "syclop mnist mix states runs "
hp.this_run_name = 'syclop_mix_'
def deploy_logs():
    if not os.path.exists(hp.save_path):
        os.makedirs(hp.save_path)

    dir_success = False
    for sfx in range(1):  # todo legacy
        candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
        if not os.path.exists(candidate_path):
            hp.this_run_path = candidate_path
            os.makedirs(hp.this_run_path)
            dir_success = True
            break
    if not dir_success:
        error('run name already exists!')

    sys.stdout = Logger(hp.this_run_path+'log.log')
    print('results are in:', hp.this_run_path)
    print('description: ', hp.description)
    #print('hyper-parameters (partial):', hp.dict)

def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp


sample = 5#int(sys.argv[4])

rnn_net = 3#int(sys.argv[1]) #Choose CNN one img network


if rnn_net == 1:
    rnn_net = no_cnn(n_timesteps = sample, input_size = 3, input_dim = 1)
elif rnn_net == 2:
    rnn_net = no_cnn_dense(n_timesteps = sample, input_size = 3, input_dim = 1)
elif rnn_net == 3:
    rnn_net = rnn_model(n_timesteps = sample, input_size = 28, input_dim = 1)
    
res = 6# int(sys.argv[3])

cnn_net = cnn_one_img(n_timesteps = sample, input_size = 28, input_dim = 1)

cnn_accur = []
cnn_train = []

rnn_accur = [] 
rnn_train = []
n_timesteps = sample
for epoch in range(50):
    deploy_logs()
    train_dataset, test_dataset = create_mnist_dataset(images, labels,res = res,
                                       sample = sample, return_datasets=True, 
                                       mixed_state = False, add_seed = 0, show_fig = False, 
                                       mix_res = True, up_sample = True, bad_res_func = bad_res101)
    train_dataset_x, train_dataset_y = mnist_split_dataset_xy(train_dataset)
    test_dataset_x, test_dataset_y = mnist_split_dataset_xy(test_dataset)
    print("##################### Fit {} and trajectories model on training data res = mix ##################".format(rnn_net.name,res))
    rnn_history = rnn_net.fit(
        train_dataset_x,
        train_dataset_y,
        batch_size=64,
        epochs=1,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch8966000080108643, 0.8992000222206116, 0.8920000195503235, 0.8925999999046326, 0.8903999924659729, 0.895600
        validation_data=(test_dataset_x, test_dataset_y),
        verbose = 0)
    print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
    rnn_accur.append(rnn_history.history['val_sparse_categorical_accuracy'][-1])
    print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])
    rnn_train.append(rnn_history.history['sparse_categorical_accuracy'][-1])
    
    print("##################### Fit {} and trajectories model on training data res = mix ##################".format(cnn_net.name,res))
    cnn_history = cnn_net.fit(
        train_dataset_x,
        train_dataset_y,
        batch_size=64,
        epochs=1,
        validation_data=(test_dataset_x, test_dataset_y),
        verbose = 0)
    print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])
    cnn_accur.append(cnn_history.history['val_sparse_categorical_accuracy'][-1])
    print('################# {} Training Accuracy = '.format(cnn_net.name),cnn_history.history['sparse_categorical_accuracy'])
    cnn_train.append(cnn_history.history['sparse_categorical_accuracy'][-1])

print('RNN Train Accuracy', rnn_train)
print('RNN Test Accuracy', rnn_accur)
print('     ')
print('CNN Train Accuracy', cnn_train)
print('CNN Test Accuracy', cnn_accur)
plt.figure()
plt.plot(rnn_train, label = 'rnn train')
plt.plot(cnn_train, label = 'cnn train')
plt.plot(rnn_accur, label = 'rnn val')
plt.plot(cnn_accur, label = 'cnn val')
plt.title('Mixed resolutions on training and res = 6 on test')
plt.legend()
plt.savefig('mixed reolution.png')