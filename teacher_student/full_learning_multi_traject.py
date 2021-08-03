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

from feature_learning_utils import  student3,  write_to_file, full_learning_dataset_update
from keras_utils import create_cifar_dataset, split_dataset_xy




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
    'num_feature' : 5,#int(sys.argv[2]),
    'num_trajectories' : 2,#int(sys.argv[3]),
    'sample' : 5,
    'res'    : 8,
    'inner_eppochs' : 1,
    'run_index' : np.random.randint(10,100),
    'dropout' : 0,
    'rnn_dropout' : 0
    }
else:
    parameters = {
    'layer_name' : layers_names[int(sys.argv[1])],
    'num_feature' : int(sys.argv[2]),
    'num_trajectories' : int(sys.argv[3]),
    'sample' : int(sys.argv[4]),
    'res'    : int(sys.argv[5]),
    'inner_eppochs' : int(sys.argv[6]),
    'run_index' : np.random.randint(10,100),
    'dropout' : 0.2,
    'rnn_dropout' : 0
    }

layer_name = parameters['layer_name']
num_feature = parameters['num_feature']
num_trajectories = parameters['num_trajectories']
sample = parameters['sample']
res = parameters['res']
inner_eppochs = parameters['inner_eppochs']
run_index = parameters['run_index']
dropout = parameters['dropout']
rnn_dropout = parameters['rnn_dropout']
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
    teacher = keras.models.load_model(path + 'cifar_trained_model')

else:
    teacher = train_model(path, trainX, trainY)

teacher.evaluate(trainX[45000:], trainY[45000:], verbose=2)


#%%
#################### Get Layer features as a dataset ##########################
print('making feature data')
intermediate_layer_model = keras.Model(inputs = teacher.input,
                                       outputs = teacher.get_layer(layer_name).output)
batch_size = 64
start = 0
end = batch_size
train_data = []
validation_data = []
train_data = np.zeros([50000,res,res,num_feature])
count = 0
#Drow N random features from the batch and sort them in order 
feature_list = np.random.choice(np.arange(64),num_feature, replace = False)
feature_list = np.sort(feature_list)

for batch in range(len(trainX)//batch_size + 1):
    count+=1
    iintermediate_output = intermediate_layer_model(trainX[start:end]).numpy()
    train_data[start:end,:,:] = iintermediate_output[:,:,:,feature_list]
    start += batch_size
    end += batch_size

      
            
   

print('loaded feature data from teacher')

#%%
feature_test_data = train_data[45000:]
feature_train_data = train_data[:45000][:,:,:]

#%%
##################### Define Student #########################################
epochs = 30
verbose = 2
evaluate_prediction_size = 150
prediction_data_path = path +'predictions/'
shape = feature_test_data.shape
teacher_mean = np.mean(feature_test_data.reshape(shape[0]*shape[1]*shape[2], shape[3]),axis = 0)
teacher_var = np.var(feature_test_data.reshape(shape[0]*shape[1]*shape[2], shape[3]),axis = 0)
print('teacher mean = ', teacher_mean, 'var =', teacher_var)
parameters['teacher_mean'] = teacher_mean
parameters['teacher_var'] = teacher_var
if num_feature == 64:
    feature_list = 'all'
parameters['feature_list'] = feature_list   
checkpoint_filepath = path + 'saved_models/{}_feature/{}_feature_net'.format(feature_list,feature_list)
save_model_path = path + 'saved_models/{}_feature/'.format(feature_list)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mean_squared_error',
    mode='min',
    save_best_only=True)


def save_model(net,path,parameters,checkpoint = True):
    home_folder = path + '{}_{}_{}_{}_{}_saved_models/'.format(feature_list, num_trajectories, sample , res, inner_eppochs)
    if not os.path.exists(home_folder):
        os.mkdir(home_folder)
    if checkpoint:
        child_folder = home_folder + 'checkpoint/'
    else:
        child_folder = home_folder + 'end_of_run_model/'
    if not os.path.exists(child_folder):
        os.mkdir(child_folder)

    #Saving weights as numpy array
    numpy_weights_path = child_folder + '{}_numpy_weights/'.format(feature_list)
    if not os.path.exists(numpy_weights_path):
        os.mkdir(numpy_weights_path)
    all_weights = net.get_weights()
    with open(numpy_weights_path + 'numpy_weights_{}_{}'.format(feature_list,num_trajectories), 'wb') as file_pi:
        pickle.dump(all_weights, file_pi)
    #LOAD WITH - pickle.load - and load manualy to model.get_layer.set_weights()
    
    #save weights with keras
    keras_weights_path = child_folder + '{}_keras_weights/'.format(feature_list)
    if not os.path.exists(keras_weights_path):
        os.mkdir(keras_weights_path)
    net.save_weights(keras_weights_path + 'keras_weights_{}_{}'.format(feature_list,num_trajectories))
    #LOADING WITH - load_status = sequential_model.load_weights("ckpt")
    

student = student3(sample = sample, 
                   res = res, 
                    activation = 'relu', 
                    dropout = dropout, 
                    rnn_dropout = rnn_dropout,
                    num_feature = num_feature)
for epoch in range(epochs):
    ######### Load Syclop Data #############
    train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                sample = sample, return_datasets=True, 
                                mixed_state = True, add_seed = num_trajectories,trajectory_list = 0
                                )
    train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample = sample)
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = sample)
    print('eval on new traject')
    student.evaluate(test_dataset_x[0],
                    feature_test_data, verbose = 2)
    print('train on new traject - {} epochs'.format(inner_eppochs))
    student_history = student.fit(train_dataset_x[0],
                    feature_train_data,
                    batch_size = 32,
                    epochs = inner_eppochs,
                    validation_data=(test_dataset_x[0], feature_test_data),
                    verbose = verbose,
                    callbacks=[model_checkpoint_callback])
    print('{} train:'.format(student.name), student_history.history['mean_squared_error'])
    print('{} test:'.format(student.name), student_history.history['val_mean_squared_error'])
save_model(student, save_model_path, parameters, checkpoint = False)
student.load_weights(checkpoint_filepath)
save_model(student, save_model_path, parameters, checkpoint = True)
student.evaluate(test_dataset_x[0],
                feature_test_data, verbose = 2)

student_test_data = np.zeros([5000,res,res,num_feature])
student_train_data = np.zeros([45000,res,res,num_feature])
start = 0
end = batch_size
count = 0
for batch in range(len(train_dataset_x[0])//batch_size + 1):
    count+=1
    train_temp = student(train_dataset_x[0][start:end]).numpy()
    student_train_data[start:end,:,:,:] = train_temp[:,:,:,:]
    start += batch_size
    end += batch_size
start = 0
end = batch_size
count = 0
for batch in range(len(test_dataset_x[0])//batch_size + 1):
    count+=1
    test_temp = student(test_dataset_x[0][start:end]).numpy()
    student_test_data[start:end,:,:,:] = test_temp[:,:,:,:]
    start += batch_size
    end += batch_size
#Evaluate per feature
var_list = []
for feature_indx in range(num_feature):
    var = np.var(student_test_data[:,:,:,feature_indx] - feature_test_data[:,:,:,feature_indx])
    var_list.append(var)
parameters['student_var'] = var_list


############################# The Student learnt the Features!! #################################################
####################### Now Let's see how good it is in classification ##########################################

#Define a Student_Decoder Network that will take the Teacher weights of the last layers:   
def Student_Decoder():
    input = keras.layers.Input(shape=(res,res,64))

    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn3')(input)
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

decoder = Student_Decoder()
layers_names = ['cnn3','cnn32','fc1','final']
#Insert the Teachers weights to the student Decoder
for layer in layers_names:
    
    teacher_weights = teacher.get_layer(layer).weights[0].numpy()
    print(teacher_weights.shape)
    print(decoder.get_layer(layer).weights[0].shape)
    new_weights = [teacher_weights, teacher.get_layer(layer).weights[1].numpy()]
    decoder.get_layer(layer).set_weights(new_weights)



################################## Sanity Check with Teachers Features ###########################################
decoder.evaluate(feature_test_data,trainY[45000:], verbose=2)

############################################## Evaluate with Student Features ###################################
print('Evaluating students features witout more training')
pre_training_accur = decoder.evaluate(student_test_data,trainY[45000:], verbose=2)
parameters['pre_training_decoder_accur'] = pre_training_accur[1]
############################ Re-train the half_net with the student training features ###########################
print('Training the base newtwork with the student features')
decoder_history = decoder.fit(student_train_data,
                       trainY[:45000],
                       epochs = 5,
                       batch_size = 64,
                       validation_data = (student_test_data, trainY[45000:]),
                       verbose = 2,)

home_folder = save_model_path + '{}_{}_{}_{}_{}_saved_models/'.format(feature_list, num_trajectories, sample , res, inner_eppochs)
decoder.save(home_folder +'decoder_trained_model')
############################## Now Let's Try and Trian the student features #####################################
########################### Combining the student and the decoder and training ##################################
print('Training the student and decoder together - reinitiating the decoder before learning')
decoder = Student_Decoder()
layers_names = ['cnn3','cnn32','fc1','final']
#Insert the Teachers weights to the student Decoder
for layer in layers_names:
    
    teacher_weights = teacher.get_layer(layer).weights[0].numpy()
    print(teacher_weights.shape)
    print(decoder.get_layer(layer).weights[0].shape)
    new_weights = [teacher_weights, teacher.get_layer(layer).weights[1].numpy()]
    decoder.get_layer(layer).set_weights(new_weights)


def full_student(student, decoder):
    input = keras.layers.Input(shape=(sample, res,res,3))\
        
    student_features = student(input)
    decoder_prediction = decoder(student_features)
    
    model = keras.models.Model(inputs=input,outputs=decoder_prediction)
    
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    
    return model 

full_student_net = full_student(student, decoder)

full_history = full_student_net.fit(train_dataset_x[0],
                       trainY[:45000],
                       epochs = 5,
                       batch_size = 64,
                       validation_data = (test_dataset_x[0], trainY[45000:]),
                       verbose = 2,)

full_student_net.save(home_folder +'full_trained_model')   
full_learning_dataset_update(student_history, decoder_history, full_history, student,parameters, name = 'multi_traject_train_{}_{}'.format(res,sample))