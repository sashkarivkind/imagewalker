#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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

from feature_learning_utils import student1, student2, student3, studentcnn, write_to_file, dataset_update,student3_cnn
from keras_utils import create_cifar_dataset, split_dataset_xy

import importlib
from misc import Logger, HP
import os 


# def deploy_logs():
    # if not os.path.exists(hp.save_path):
    #     os.makedirs(hp.save_path)

    # dir_success = False
    # for sfx in range(1):  # todo legacy
    #     candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
    #     if not os.path.exists(candidate_path):
    #         hp.this_run_path = candidate_path
    #         os.makedirs(hp.this_run_path)
    #         dir_success = True
    #         break
    # if not dir_success:
    #     print('run name already exists!')

    # sys.stdout = Logger(hp.this_run_path+'log.log')
    # print('results are in:', hp.this_run_path)
    # print('description: ', hp.description)
    #print('hyper-parameters (partial):', hp.dict)
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

# hp.description = "syclop micro feature learning runs"
# hp.this_run_name = 'micro_{}'.format(run_index)
# deploy_logs()
#%%
############################### Get Trained Teacher ##########################3
if len(sys.argv) == 1:
    path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'
else: 
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
    model = keras.models.load_model(path + 'cifar_trained_model2')

else:
    model = train_model(path, trainX, trainY)

model.evaluate(trainX[45000:], trainY[45000:])


#%%
#################### Get Layer features as a dataset ##########################
#feature_data_path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/feature_data/
#feature_data_path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_data/'
# feature_data_path = path +'feature_data/'
# if os.path.exists(feature_data_path + 'train_features_{}'.format(layer_name)):
#     print('found data')
#     train_data = np.array(pickle.load(open(feature_data_path + 'train_features_{}'.format(layer_name),'rb')))
#     #test_data = np.array(pickle.load(open('/home/orram/Documents/GitHub/imagewalker/teacher_student/feature_data/test_features_{}'.format(layer_name),'rb')))
# '''
# else:
print('making feature data')
intermediate_layer_model = keras.Model(inputs = model.input,
                                       outputs = model.get_layer('max_pool2').output)
batch_size = 64
start = 0
end = batch_size
train_data = []
validation_data = []
train_data = np.zeros([50000,8,8,64])
for batch in range(len(trainX)//batch_size + 1):
    iintermediate_output = intermediate_layer_model(trainX[start:end]).numpy()
    train_data[start:end,:,:] = iintermediate_output[:,:,:,:]
    # iintermediate_output = list(intermediate_layer_model(testX[start:end]))
    # validation_data += iintermediate_output
    start += batch_size
    end += batch_size
# with open(feature_data_path + 'train_features_{}'.format(layer_name), 'wb') as file_pi:
#     pickle.dump(train_data, file_pi)
# with open(feature_data_path + 'validation_features_{}'.format(layer_name), 'wb') as file_pi:
#     pickle.dump(validation_data, file_pi)
#train_data = np.array(train_data)

print('loaded feature data from teacher')
#%%
test_data = train_data[45000:]
#%%
feature_test_data = test_data[:,:,:]
feature_train_data = train_data[:45000][:,:,:]
#%%

############################## load syclop data #################################
# print('loading Syclop Data')
sample = 10
# traject_data_path = path +'traject_data/'
# if load_saved_traject:
#     print('load_saved_traject')
#     with open(traject_data_path + 'traject_{}_train'.format(load_saved_traject), 'rb') as file_pi:
#            (train_dataset_x,train_dataset_y) = pickle.load(file_pi)
#     with open(traject_data_path + 'traject_{}_test'.format(load_saved_traject), 'rb') as file_pi:
#             (test_dataset_x,test_dataset_y) = pickle.load(file_pi)
# '''
# else:
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = 8,
                                sample = sample, return_datasets=True, 
                                mixed_state = False, add_seed = 0,trajectory_list = trajectory_index
                                )
train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample = sample)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = sample)
#print('saving trajectory data')
    #traject_data_path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/traject_data/'
    # traject_data_path = path +'traject_data/'
    # with open(traject_data_path + 'traject_{}_train'.format(run_index), 'wb') as file_pi:
    #         pickle.dump((train_dataset_x,train_dataset_y), file_pi)
    # with open(traject_data_path + 'traject_{}_test'.format(run_index), 'wb') as file_pi:
    #         pickle.dump((test_dataset_x,test_dataset_y), file_pi)
# '''
# with open('/home/orram/Documents/GitHub/imagewalker/teacher_student/train_features_{}'.format(layer_name), 'wb') as file_pi:
#         pickle.dump(train_data, file_pi)
# with open('/home/orram/Documents/GitHub/imagewalker/teacher_student/test_features_{}'.format(layer_name), 'wb') as file_pi:
#         pickle.dump(test_data, file_pi)
#%%
##################### Define Student #########################################
epochs = 1
verbose = 1
evaluate_prediction_size = 150
prediction_data_path = path +'predictions/'
teacher_mean = np.mean(feature_test_data)
teacher_var = np.var(feature_test_data)
parameters['teacher_mean'] = teacher_mean
parameters['teacher_var'] = teacher_var
def train_student(student, teacher, ):
    
    student.evaluate(test_dataset_x[0],
                    feature_test_data)
    student_history = student.fit(train_dataset_x[0],
                    feature_train_data,
                    batch_size = 32,
                    epochs = epochs,
                    validation_data=(test_dataset_x[0], feature_test_data),
                    verbose = verbose)
    print('{} train:'.format(student.name), student_history.history['mean_squared_error'])
    print('{} test:'.format(student.name), student_history.history['val_mean_squared_error'])
    #dataset_update(student_history, student,parameters)
    
    student_predictions = student(test_dataset_x[0][:evaluate_prediction_size])
    
    return student_predictions


# student1_predictions = train_student(student1(), model, )
# student2_predictions = train_student(student2(), model, )
student3_predictions = train_student(student3(), model, )
# student_cnn_predictions = train_student(studentcnn(), model, )

with open(prediction_data_path + 'predictions_2_{}_{}_{}'.format(layer_name, feature, run_index), 'wb') as file_pi:
    pickle.dump((feature_test_data[:evaluate_prediction_size], student3_predictions), file_pi) 




