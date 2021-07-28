#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from datetime import datetime
import os 
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

if tf.test.gpu_device_name(): 
    print(len(tf.config.list_physical_devices('GPU')))
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:
    print("no gpu working")
    
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

import pickle

from feature_learning_utils import student1, student2, student32, studentcnn, write_to_file, dataset_update
from keras_utils import create_cifar_dataset, split_dataset_xy,print_traject

import importlib
from misc import Logger, HP
import os 


# def deploy_logs():
#     if not os.path.exists(hp.save_path):
#         os.makedirs(hp.save_path)

#     dir_success = False
#     for sfx in range(1):  # todo legacy
#         candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
#         hp.this_run_path = candidate_path
#         os.makedirs(hp.this_run_path)
#         dir_success = True
#         break
#     if not dir_success:
#         print('run name already exists!')

#     sys.stdout = Logger(hp.this_run_path+'log.log')
#     print('results are in:', hp.this_run_path)
#     print('description: ', hp.description)
#     #print('hyper-parameters (partial):', hp.dict)
print_traject(images, labels,res = 8,
                                sample = sample, return_datasets=True, 
                                mixed_state = False, add_seed = 0, trajectory_list = 40,
                                )
#%%

layers_names = [
    'input_1',
    'cnn1',
    'cnn12',
    "max_pool1",
    'cnn2',
    'cnn22',
    'max_pool2',
    'cnn3',
    'cnn32',
    'max_pool3',
    'fc1',
    'final',
    ]
# load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
images, labels = trainX, trainY


if len(sys.argv) == 1:
    parameters = {
    'layer_name' : 'cnn32',#layers_names[int(sys.argv[1])],
    'feature' : 86,#int(sys.argv[2]),
    'trajectory_index' : 40,#int(sys.argv[3]),
    'run_index' : np.random.randint(10,100),
    }
else:
    parameters = {
    'layer_name' : layers_names[int(sys.argv[1])],
    'feature' : int(sys.argv[2]),
    'trajectory_index' : int(sys.argv[3]),
    'run_index' : np.random.randint(10,100),
    }

layer_name = parameters['layer_name']
feature = parameters['feature']
trajectory_index = parameters['trajectory_index']
run_index = parameters['run_index']
if not trajectory_index:
    trajectory_index = run_index
print(parameters)
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

# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)

# hp = HP()
# hp.save_path = 'saved_runs'

# hp.description = "syclop micro feature learning rutrain_dataset, test_dataset = create_cifar_dataset(images, labels,res = 8,
                                sample = sample, return_datasets=True, 
                                mixed_state = False, add_seed = 0, trajectory_list = trajectory_index,
                                )ns"
# hp.this_run_name = 'micro_{}'.format(run_index)
# deploy_logs()

class feature_st(keras.Model):
    def __init__(self, student, teacher, layer_name):
        super(feature_st, self).__init__()
        self.teacher = teacher
        self.layer_name = layer_name
        self.intermediate_layer_model = keras.Model(inputs = self.teacher.input,
                                       outputs = model.get_layer(self.layer_name).output)
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        #student_loss_fn,
        features_loss_fn,
        feature,
        # beta=0.1,
        # temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(feature_st, self).compile(optimizer=optimizer, metrics=metrics)
        #self.student_loss_fn = student_loss_fn
        self.features_loss_fn = features_loss_fn
        self.feature = feature
        self.teacher_mean = []
        self.teacher_std = []
        # self.beta = beta
        # self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        syclop_data, HR_data = data
        # Forward pass of teacher
        teacher_features = self.intermediate_layer_model(HR_data, training=False)[:, : , :, self.feature]

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_features = self.student(syclop_data, training=True)

            # Compute losses
            #student_loss = self.student_loss_fn(y, student_predictions)
            features_loss = self.features_loss_fn(
                teacher_features,
                student_features,
            )
            loss = features_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(teacher_features, student_features)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {'features_loss': features_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        syclop_data, HR_data = data
        teacher_features = self.intermediate_laytrain_dataset, test_dataset = create_cifar_dataset(images, labels,res = 8,
                                sample = sample, return_datasets=True, 
                                mixed_state = False, add_seed = 0, trajectory_list = trajectory_index,
                                )er_model(HR_data, training=False)[:, : , :, self.feature]
        
        shape = teacher_features.shape
        self.teacher_mean.append(keras.backend.mean(teacher_features))
        self.teacher_std.append(keras.backend.mean(keras.backend.std(keras.backend.flatten(teacher_features))))
        
        # Compute predictions
        y_prediction = self.student(syclop_data, training=False)
        # Calculate the loss
        features_loss = self.features_loss_fn(teacher_features, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(teacher_features, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"feature_loss": features_loss})
        return results
    
    def get_teacher_std(self, data):
        HR_data = data
        teacher_features = self.intermediate_layer_model(HR_data, training=False)[:, : , :, self.feature]
        
        shape = teacher_features.shape
        teacher_mean = keras.backend.mean(teacher_features).numpy()
        teacher_std = keras.backend.mean(keras.backend.std(keras.backend.flatten(teacher_features))).numpy()
        self.teacher_mean.append(teacher_mean)
        self.teacher_std.append(teacher_std)
        
        
        return teacher_mean, teacher_std

############################### Get Trained Teacher ##########################3
path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'
path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/'
def train_model(path, trainX, trainY):
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
        #Flatten and add linear layer and softmax
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
    
    
    # define model
    model = net()

    history = model.fit(trainX[:45000], 
                        trainY[:45000], 
                        epochs=15, 
                        batch_size=64, 
                        validation_data=(trainX[45000:], trainY[45000:]), 
                        verbose=0)
    
    #Save Network
    model.save(path +'cifar_trained_model')
    
    #plot_results
    plt.figure()
    plt.plot(history.history['sparse_categorical_accuracy'], label = 'train')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'test')
    plt.legend()
    plt.grid()
    plt.title('Cifar10 - train/test accuracies')
    plt.savefig('Saved Networks accur plot')
    

    return model 
    
if os.path.exists(path + 'cifar_trained_model'):
    model = keras.models.load_model(path + 'cifar_trained_model')

#else:
#    model = train_model(path, trainX, trainY)

#model.evaluate(trainX[45000:], trainY[45000:])


#%%#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:59:31 2021

@author: orram
"""


############################## load syclop data #################################
print('loading Syclop Data')
sample = 10
traject_data_path = path +'traject_data/'
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = 8,
                                sample = sample, return_datasets=True, 
                                mixed_state = False, add_seed = 0, trajectory_list = trajectory_index,
                                )

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample = sample)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = sample)
print('loaded trajectory data')
#traject_data_path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/traject_data/'
traject_data_path = path +'traject_data/'



#%%
##################### Define Student #########################################
epochs = 1
verbose = 2
prediction_data_path = path +'predictions/'

print('##################### learning feature {} ########################'.format(feature))
intermediate_layer_model = keras.Model(inputs = model.input,
                                       outputs = model.get_layer(layer_name).output)
target = intermediate_layer_model(trainX[45000 + 50:45000 + 150])[:,:,:,feature]

# student_net1 = student1()
# feature_learning = feature_st(student_net1, model, layer_name)
# before = student_net1(test_dataset_x[0][2:5]) * 1
# feature_learning.compile(optimizer = keras.optimizers.Adam(lr = 1e-3),
#            metrics   = ["mean_squared_error"],
#            features_loss_fn = keras.losses.MeanSquaredError(),
#            feature = feature)
# student_history1 = feature_learning.fit(train_dataset_x[0],
#                 trainX[:45000],
#                 batch_size = 32,
#                 epochs = epochs,
#                 validation_data=(test_dataset_x[0], trainX[45000:]),
#                 verbose = verbose)
# print('student1 net train:', student_history1.history['mean_squared_error'])
# print('student1 net test:', student_history1.history['val_mean_squared_error'])
# teacher_mean, teacher_std = feature_learning.get_teacher_std(trainX[45000:])
# parameters['teacher_mean'] = teacher_mean
# parameters['teacher_std'] = teacher_std
# print('teacher feature mean =', teacher_mean, 'teacher feature std =', teacher_std )
# student1_prediction = student_net1(test_dataset_x[0][50:150]) * 1
# # write_to_file(student_history1, student_net1,parameters)
# dataset_update(student_history1, student_net1,parameters)
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))
student_net2 = student2()
feature_learning2 = feature_st(student_net2, model, layer_name)
feature_learning2.compile(optimizer = keras.optimizers.Adam(lr = 1e-3),
           metrics   = ["mean_squared_error"],
           features_loss_fn = keras.losses.MeanSquaredError(),
           feature = feature)
student_history2 = feature_learning2.fit(train_dataset_x[0],
                trainX[:45000],
                batch_size = 32,
                epochs = epochs,
                validation_data=(test_dataset_x[0], trainX[45000:]),
                verbose = verbose)
print('student2 net train:', student_history2.history['mean_squared_error'])
print('student2 net test:', student_history2.history['val_mean_squared_error'])
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))
student2_prediction = student_net2(test_dataset_x[0][50:150]) * 1
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))
# write_to_file(student_history2, student_net2,parameters)
dataset_update(student_history2, student_net2,parameters)
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))

student_net3 = student3()
feature_learning3 = feature_st(student_net3, model, layer_name)
feature_learning3.compile(optimizer = keras.optimizers.Adam(lr = 1e-3),
           metrics   = ["mean_squared_error"],
           features_loss_fn = keras.losses.MeanSquaredError(),
           feature = 42)
student_history3 = feature_learning3.fit(train_dataset_x[0],
                trainX[:45000],
                batch_size = 32,
                epochs = epochs,
                validation_data=(test_dataset_x[0], trainX[45000:]),
                verbose = verbose)
print('student3 net train:', student_history3.history['mean_squared_error'])
print('student3 net test:', student_history3.history['val_mean_squared_error'])
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))
student3_prediction = student_net3(test_dataset_x[0][50:150]) * 1
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))
# write_to_file(student_history3, student_net3,parameters)
dataset_update(student_history3, student_net3,parameters)
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))

# student_net_cnn = studentcnn()
# feature_learning_cnn = feature_st(student_net_cnn, model, layer_name)
# feature_learning_cnn.compile(optimizer = keras.optimizers.Adam(lr = 1e-3),
#            metrics   = ["mean_squared_error"],
#            features_loss_fn = keras.losses.MeanSquaredError(),
#            feature = 42)
# student_history_cnn = feature_learning_cnn.fit(train_dataset_x[0],
#                 trainX[:45000],
#                 batch_size = 32,
#                 epochs = epochs,
#                 validation_data=(test_dataset_x[0], trainX[45000:]),
#                 verbose = verbose)
# print('student-cnn net train:', student_history_cnn.history['mean_squared_error'])
# print('student-cnn net test:', student_history_cnn.history['val_mean_squared_error'])
# student_cnn_prediction = student_net_cnn(test_dataset_x[0][50:150]) * 1
# # write_to_file(student_history_cnn, student_net_cnn,parameters)
# dataset_update(student_history_cnn, student_net_cnn,parameters)

    
    
    

with open(prediction_data_path + 'predictions_{}_{}_{}'.format(layer_name, feature, run_index), 'wb') as file_pi:
    pickle.dump((target, student2_prediction,student3_prediction, ), file_pi) #student1_prediction,student_cnn_prediction
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))


