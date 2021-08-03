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
half_net.evaluate(feature_test_data,trainY[45000:], verbose=2)

############################################ Get Syclop Data ######################################################
print('Loading Syclop Data with trajectory index {}'.format(trajectory_index))
from keras_utils import create_cifar_dataset, split_dataset_xy
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = 8,
                                sample = 10, return_datasets=True, 
                                mixed_state = False, add_seed = 0,trajectory_list = trajectory_index
                                )
train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample = 10)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = 10)

#%%
################################# Now, let's extract the trianing features      ##################################
################################## And let the network relearn from our features ##################################
################################# Extract Feature from Learnt Student #############################################
print('Extracting Student Feature from Trained Networks')
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker/teacher_student')
from feature_learning_utils import student3
path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/'
path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'
student_test_data = np.zeros([5000,8,8,64])
student_train_data = np.zeros([45000,8,8,64])
t_f = True
feature_list = np.random.choice(np.arange(64),64, replace = False)
feature_list = np.sort(feature_list)

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
start = 0
end = batch_size
count = 0
for batch in range(len(train_dataset_x[0])//batch_size + 1):
    count+=1
    train_temp = numpy_student(train_dataset_x[0][start:end]).numpy()
    student_train_data[start:end,:,:,:] = train_temp
    start += batch_size
    end += batch_size
start = 0
end = batch_size
count = 0
for batch in range(len(test_dataset_x[0])//batch_size + 1):
    count+=1
    test_temp = numpy_student(test_dataset_x[0][start:end]).numpy()
    student_test_data[start:end,:,:, :] = test_temp
    start += batch_size
    end += batch_size



############################################## Evaluate with Student Features ###################################
print('Evaluating students features witout more training')
half_net.evaluate(student_test_data,trainY[45000:], verbose=1)

############################ Re-train the half_net with the student training features ###########################
print('Training the base newtwork with the student features')
history = half_net.fit(student_train_data,
                       trainY[:45000],
                       epochs = 5,
                       batch_size = 64,
                       validation_data = (student_test_data, trainY[45000:]),
                       verbose = 1,)
                        

#Save Network
half_net.save(path +'student_half_net_trained')
# prediction_data_path = path +'predictions/'
# with open(prediction_data_path + 'predictions_traject_{}_{}_{}_{}'.format('all_layers', feature, trajectory_index,run_index,), 'wb') as file_pi:
#     pickle.dump((student_train_data, student_test_data), file_pi)

#%%
############################## Now Let's Try and Trian the student features #####################################
########################### Combining the student and the decoder and training ##################################

def full_student(student, half_net):
    input = keras.layers.Input(shape=(10, 8,8,3))\
        
    student_features = student(input)
    decoder_prediction = half_net(student_features)
    
    model = keras.models.Model(inputs=input,outputs=decoder_prediction)
    
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    
    return(model)

full_student_net = full_student(numpy_student, half_net)

full_history = full_student_net.fit(train_dataset_x[0],
                       trainY[:45000],
                       epochs = 1,
                       batch_size = 64,
                       validation_data = (test_dataset_x[0], trainY[45000:]),
                       verbose = 1,)
    

 