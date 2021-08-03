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
train_data, test_data = prep_pixels(trainX, testX)
images, labels = testX, testY

#%%
parameters = {
'layer_name' : 'max_pool2',#layers_names[int(sys.argv[1])],
'trajectory_index' : 42,#int(sys.argv[3]),
'run_index' : np.random.randint(100,1000),
'dropout' : 0.2,
'rnn_dropout' : 0
}


layer_name = parameters['layer_name']
trajectory_index = parameters['trajectory_index']
run_index = parameters['run_index']
dropout = parameters['dropout']
rnn_dropout = parameters['rnn_dropout']
print(parameters)
path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/'
path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'


teacher = keras.models.load_model(path + 'cifar_trained_model')
teacher.evaluate(testX, testY, verbose=2)


########################### Network that takes feature space as input ############################################
########################### With the same weights as the teacher      ############################################
def half_teacher():
    input = keras.layers.Input(shape=(8,8,64))

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

if os.path.exists(path + 'student_half_net_trained'):
    print('found trained decoder')
    half_net = keras.models.load_model(path + 'student_half_net_trained')
else:
    half_net = half_teacher()
    layers_names = ['cnn3','cnn32','fc1','final']
    for layer in layers_names:
        
        teacher_weights = teacher.get_layer(layer).weights[0].numpy()
        print(teacher_weights.shape)
        print(half_net.get_layer(layer).weights[0].shape)
        new_weights = [teacher_weights, teacher.get_layer(layer).weights[1].numpy()]
        half_net.get_layer(layer).set_weights(new_weights)


############################################ Get Syclop Data ######################################################
print('Loading Syclop Data with trajectory index {}'.format(trajectory_index))
from keras_utils import create_cifar_dataset, split_dataset_xy
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = 8,
                                sample = 10, return_datasets=True, 
                                mixed_state = False, add_seed = 0,trajectory_list = trajectory_index
                                )
test_dataset_x, test_dataset_y = split_dataset_xy(train_dataset, sample = 10)


#%%
################################# Now, let's extract the trianing features      ##################################
################################## And let the network relearn from our features ##################################
################################# Extract Feature from Learnt Student #############################################
print('Extracting Student Feature from Trained Networks')
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker/teacher_student')
from feature_learning_utils import student3
path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/'
path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'

t_f = True
feature_list = 'all'
temp_path = path + 'saved_models/{}_feature/'.format(feature_list)
home_folder = temp_path + '{}_{}_saved_models/'.format(feature_list, trajectory_index)
checkpoint = t_f
if checkpoint:
    child_folder = home_folder + 'checkpoint/'
else:
    child_folder = home_folder + 'end_of_run_model/'


#loading weights as numpy array
numpy_weights_path = child_folder + '{}_numpy_weights/'.format(feature_list)
with open(numpy_weights_path + 'numpy_weights_{}_{}'.format(feature_list,trajectory_index), 'rb') as file_pi:
    np_weights = pickle.load(file_pi)
numpy_student = student3(activation = 'relu', dropout = 0.2, rnn_dropout = 0, num_feature = 64)
layer_index = 0
for i in range(3):
    layer_name = 'convLSTM{}'.format(i+1)
    saved_weights = [np_weights[layer_index], np_weights[layer_index+ 1], np_weights[layer_index+ 2]]
    numpy_student.get_layer(layer_name).set_weights(saved_weights)
    layer_index += 3
res = 8
sample = 10
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
    
    return(model)

full_student_net = full_student(numpy_student, half_net)

############################################## Evaluate with Student Features ###################################
#%%
print('Evaluating students features witout more training')
full_student_net.evaluate(test_dataset_x[0],testY, verbose=1)

#%%
print('Loading Syclop Data with trajectory index {}'.format(trajectory_index))
from keras_utils import create_cifar_dataset, split_dataset_xy
train_dataset, test_dataset = create_cifar_dataset(trainX, trainY,res = 8,
                                sample = 10, return_datasets=True, 
                                mixed_state = False, add_seed = 0,trajectory_list = trajectory_index
                                )
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset, sample = 10)

full_student_net.evaluate(test_dataset_x[0],test_dataset_y, verbose=1)
