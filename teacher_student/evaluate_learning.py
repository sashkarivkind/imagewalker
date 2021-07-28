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
images, labels = trainX, trainY
# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)

path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/'
#path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'
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

teacher = keras.models.load_model(path + 'cifar_trained_model')
teacher.evaluate(trainX[45000:], trainY[45000:], verbose=2)

print('making teacher feature data')
intermediate_layer_model = keras.Model(inputs = teacher.input,
                                       outputs = teacher.get_layer('max_pool2').output)
batch_size = 64
start = 0
end = batch_size
train_data = []
validation_data = []
train_data = np.zeros([50000,8,8,64])
count = 0
state_feature = False
while state_feature is False:
    if count > 10:
        break
    for batch in range(len(trainX)//batch_size + 1):
        count+=1
        iintermediate_output = intermediate_layer_model(trainX[start:end]).numpy()
        train_data[start:end,:,:] = iintermediate_output
        # iintermediate_output = list(intermediate_layer_model(testX[start:end]))
        # validation_data += iintermediate_output
        start += batch_size
        end += batch_size


feature_test_data = train_data[45000:]
feature_train_data = train_data[:45000]

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

half_net = half_teacher()
layers_names = ['cnn3','cnn32','fc1','final']
for layer in layers_names:
    
    teacher_weights = teacher.get_layer(layer).weights[0].numpy()
    print(teacher_weights.shape)
    print(half_net.get_layer(layer).weights[0].shape)
    new_weights = [teacher_weights, teacher.get_layer(layer).weights[1].numpy()]
    half_net.get_layer(layer).set_weights(new_weights)



################################## Sanity Check with Teachers Features ###########################################
half_net.evaluate(feature_test_data,trainY[45000:], verbose=1)

################################# Insert our models features and pray #############################################
#%%
files_path = path + 'predictions/'

files = os.listdir(files_path)
prediction_data = np.zeros([5000,8,8,64])
teacher_data = np.zeros([5000,8,8,64])
for f in files:
    if len(f) > 23:
        if f[:23] == 'predictions_4_max_pool2':
            if len(f) == 29:
                feature_num = int(f[24:26])
                data = pickle.load(open(files_path + f,'rb'))
                prediction_data[:,:,:,feature_num] = np.reshape(data[1], (5000,8,8))
                teacher_data[:,:,:,feature_num] = data[0]
            else:
                feature_num = int(f[24])
                data = pickle.load(open(files_path + f,'rb'))
                prediction_data[:,:,:,feature_num] = np.reshape(data[1], (5000,8,8))
                teacher_data[:,:,:,feature_num] = data[0]
                
            
	

################################## Sanity Check #2 with Teachers Features ########################################
################################## Extracted from training saved data ############################################
half_net.evaluate(teacher_data,trainY[45000:], verbose=1)

################################## Now Testing on our reconstructed data  ########################################
################################## Extracted from training saved data ############################################
half_net.evaluate(prediction_data,trainY[45000:], verbose=1)
#%%
############################################ Get Syclop Data ######################################################
from keras_utils import create_cifar_dataset, split_dataset_xy
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = 8,
                                sample = 10, return_datasets=True, 
                                mixed_state = False, add_seed = 0,trajectory_list = 2
                                )
train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample = 10)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = 10)
#%%
################################## Now, let's extract the trianing features      ##################################
################################## And let the network relearn from our features ##################################
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker/teacher_student')
from feature_learning_utils import student3
for feature in range(4):
    checkpoint_filepath = path + 'saved_models/{}_feature/{}_feature_net'.format(feature,feature)
    temp_student = student3(activation = 'relu', dropout = 0.2, rnn_dropout = 0)
    temp_student.evaluate(test_dataset_x[0],
                    feature_test_data, verbose = 1)    
    temp_student.load_weights(checkpoint_filepath).expect_partial()
    temp_student.evaluate(test_dataset_x[0],
                    feature_test_data, verbose =2)

np.random.seed(2)
np.random.randint(0,10,10)