#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:41:43 2021

@author: orram
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
added the explore relations part after 735561
"""

import os 
import sys
import gc
sys.path.insert(1, '/home/labs/ahissarlab/arivkind/imagewalker')
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
import time
import pickle
import argparse
from feature_learning_utils import load_student, student3,  write_to_file, traject_learning_dataset_update,  net_weights_reinitializer
from keras_utils import create_cifar_dataset, split_dataset_xy

print(os.getcwd() + '/')
#%%

# load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
images, labels = trainX, trainY



parser = argparse.ArgumentParser()

#general parameters
parser.add_argument('--run_name_prefix', default='noname', type=str, help='path to pretrained teacher net')
parser.add_argument('--run_index', default=10, type=int, help='run_index')

parser.add_argument('--testmode', dest='testmode', action='store_true')
parser.add_argument('--no-testmode', dest='testmode', action='store_false')


### teacher network parameters
parser.add_argument('--teacher_net', default='/home/orram/Documents/GitHub/imagewalker/teacher_student/model_510046__1628691784.hdf', type=str, help='path to pretrained teacher net')

parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
parser.add_argument('--last_layer_size', default=128, type=int, help='last_layer_size')


parser.add_argument('--dropout1', default=0.2, type=float, help='dropout1')
parser.add_argument('--dropout2', default=0.0, type=float, help='dropout2')
parser.add_argument('--dataset_norm', default=128.0, type=float, help='dropout2')
parser.add_argument('--dataset_center', dest='dataset_center', action='store_true')
parser.add_argument('--no-dataset_center', dest='dataset_center', action='store_false')


parser.add_argument('--layer_norm_res', dest='layer_norm_res', action='store_true')
parser.add_argument('--no-layer_norm_res', dest='layer_norm_res', action='store_false')

parser.add_argument('--layer_norm_2', dest='layer_norm_2', action='store_true')
parser.add_argument('--no-layer_norm_2', dest='layer_norm_2', action='store_false')

parser.add_argument('--skip_conn', dest='skip_conn', action='store_true')
parser.add_argument('--no-skip_conn', dest='skip_conn', action='store_false')

parser.add_argument('--last_maxpool_en', dest='last_maxpool_en', action='store_true')
parser.add_argument('--no-last_maxpool_en', dest='last_maxpool_en', action='store_false')

parser.add_argument('--nl', default='relu', type=str, help='non linearity')

parser.add_argument('--stopping_patience', default=10, type=int, help='stopping patience')
parser.add_argument('--learning_patience', default=5, type=int, help='stopping patience')
parser.add_argument('--manual_suffix', default='', type=str, help='manual suffix')

parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
parser.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false')

parser.add_argument('--rotation_range', default=0.0, type=float, help='dropout1')
parser.add_argument('--width_shift_range', default=0.1, type=float, help='dropout2')
parser.add_argument('--height_shift_range', default=0.1, type=float, help='dropout2')

parser.set_defaults(data_augmentation=True,layer_norm_res=True,layer_norm_student=True,layer_norm_2=True,skip_conn=True,last_maxpool_en=True, testmode=False,dataset_center=True)

config = parser.parse_args()
config = vars(config)
print('config  ',config)

parameters = config
TESTMODE = parameters['testmode']


lsbjob = os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob

this_run_name = parameters['run_name_prefix'] + '_j' + lsbjob + '_t' + str(int(time.time()))
parameters['this_run_name'] = this_run_name


print(parameters)
# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    #center
    if parameters['dataset_center']:
        mean_image = np.mean(train_norm, axis=0)
        train_norm -= mean_image
        test_norm -= mean_image
    # normalize to range 0-1
    train_norm = train_norm / parameters['dataset_norm']
    test_norm = test_norm /  parameters['dataset_norm']
    # return normalized images
    return train_norm, test_norm

# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)


#%%
############################### Get Trained Teacher ##########################3

path = os.getcwd() + '/'

teacher = keras.models.load_model(parameters['teacher_net'])
#teacher.evaluate(trainX[45000:], trainY[45000:], verbose=2)


fe_model = teacher.layers[0]
be_model = teacher.layers[1]

#%%
student, parameters = load_student()
res = parameters['res']
mode = 'from_predictions'
if mode =='with_student':
    train_dataset, test_dataset, seed_list = create_cifar_dataset(images, labels,res = res,
                                sample = parameters['sample'], return_datasets=True, 
                                mixed_state = True, 
                                add_seed = parameters['trajectories_num'],
                                trajectory_list = 0,
                                broadcast=parameters['broadcast'],
                                style = parameters['style'],
                                max_length=parameters['max_length'], 
                                noise = parameters['noise'],
                                )
    train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample = parameters['sample'])
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = parameters['sample'])
    del train_dataset
    del test_dataset
    gc.collect()
   
    student_test_data = np.zeros([5000,res,res,64])
    student_train_data = np.zeros([45000,res,res,64])
    batch_size = 64
    start = 0
    end = batch_size
    count = 0
    print('\nExtracting student learnt features')
    for batch in range(len(train_dataset_x[0])//batch_size + 1):
        count+=1
        train_temp = student((train_dataset_x[0][start:end],train_dataset_x[1][start:end])).numpy()
        student_train_data[start:end,:,:,:] = train_temp[:,:,:,:]
        start += batch_size
        end += batch_size
    start = 0
    end = batch_size
    count = 0
    for batch in range(len(test_dataset_x[0])//batch_size + 1):
        count+=1
        test_temp = student((test_dataset_x[0][start:end],test_dataset_x[1][start:end])).numpy()
        student_test_data[start:end,:,:,:] = test_temp[:,:,:,:]
        start += batch_size
        end += batch_size
elif mode == 'from_predictions':
    pred_name = 'predictions_traject_noname_j178_t1630240486'
    pred_path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/predictions/'
    data = pickle.load(open(pred_path + pred_name,'rb'))
    student_test_data = data[1]
    student_train_data = data[3]
#%%

decoder =  keras.models.clone_model(be_model)
opt=tf.keras.optimizers.Adam(lr=1e-3)
decoder.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

################################## Evaluate with Student Features ###################################
print('\nEvaluating students features witout more training')
lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=1, min_lr=0.5e-6)
early_stopper = keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy', min_delta=1e-4, patience=5, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)
pre_training_accur = decoder.evaluate(student_test_data,trainY[45000:], verbose=2)

############################ Re-train the half_net with the student training features ###########################
print('\nTraining the base newtwork with the student features')
decoder_history = decoder.fit(student_train_data,
                       trainY[:45000],
                       epochs = 3  if not TESTMODE else 1,
                       batch_size = 64,
                       validation_data = (student_test_data, trainY[45000:]),
                       verbose = 1,
                       callbacks=[lr_reducer,early_stopper],)
    
    
#%% ############################ Add regularization ###########################
# adding regularization
decoder =  keras.models.clone_model(be_model)
regularizer = tf.keras.regularizers.l2()

for layer in decoder.layers:
    for attr in ['kernel_regularizer']:
        if hasattr(layer, attr):
          setattr(layer, attr, regularizer)
for layer in decoder.layers:
    if layer.name[:3] == 'cnn':
        layer_name = layer.name
        saved_weights = [np_weights[layer_index], np_weights[layer_index+ 1], np_weights[layer_index+ 2]]
        numpy_student.get_layer(layer_name).set_weights(saved_weights)
        layer_index += 3
opt=tf.keras.optimizers.Adam(lr=1e-3)
decoder.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

print('\nTraining the base newtwork with the student features')
decoder_history = decoder.fit(student_train_data,
                       trainY[:45000],
                       epochs = 5  if not TESTMODE else 1,
                       batch_size = 64,
                       validation_data = (student_test_data, trainY[45000:]),
                       verbose = 1,
                       callbacks=[lr_reducer,early_stopper],)
