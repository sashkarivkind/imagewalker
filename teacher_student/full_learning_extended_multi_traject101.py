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
from feature_learning_utils import  student3,  write_to_file, traject_learning_dataset_update,  net_weights_reinitializer
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

### student parameters
parser.add_argument('--epochs', default=1, type=int, help='num training epochs')
parser.add_argument('--int_epochs', default=1, type=int, help='num internal training epochs')
parser.add_argument('--decoder_epochs', default=20, type=int, help='num internal training epochs')
parser.add_argument('--num_feature', default=64, type=int, help='legacy to be discarded')
parser.add_argument('--rnn_layer1', default=32, type=int, help='legacy to be discarded')
parser.add_argument('--rnn_layer2', default=64, type=int, help='legacy to be discarded')
parser.add_argument('--time_pool', default=0, help='time dimention pooling to use - max_pool, average_pool, 0')

parser.add_argument('--student_block_size', default=1, type=int, help='number of repetition of each convlstm block')
parser.add_argument('--conv_rnn_type', default='lstm', type=str, help='conv_rnn_type')
parser.add_argument('--student_nl', default='relu', type=str, help='non linearity')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout1')
parser.add_argument('--rnn_dropout', default=0.0, type=float, help='dropout1')
conv_rnn_type='lstm'

parser.add_argument('--layer_norm_student', dest='layer_norm_student', action='store_true')
parser.add_argument('--no-layer_norm_student', dest='layer_norm_student', action='store_false')


### syclop parameters
parser.add_argument('--trajectory_index', default=0, type=int, help='trajectory index - set to 0 because we use multiple trajectories')
parser.add_argument('--sample', default=5, type=int, help='sample')
parser.add_argument('--res', default=8, type=int, help='resolution')
parser.add_argument('--trajectories_num', default=10, type=int, help='number of trajectories to use')
parser.add_argument('--broadcast', default=0, type=int, help='1-integrate the coordinates by broadcasting them as extra dimentions, 2- add coordinates as an extra input')
parser.add_argument('--style', default='brownain', type=str, help='choose syclops style of motion')
parser.add_argument('--noise', default=0.15, type=float, help='added noise to the const_p_noise style')
parser.add_argument('--max_length', default=5, type=int, help='choose syclops max trajectory length')


### teacher network parameters
parser.add_argument('--teacher_net', default='/home/orram/Documents/GitHub/imagewalker/teacher_student/model_510046__1628691784.hdf', type=str, help='path to pretrained teacher net')

parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
parser.add_argument('--last_layer_size', default=128, type=int, help='last_layer_size')


parser.add_argument('--dropout1', default=0.2, type=float, help='dropout1')
parser.add_argument('--dropout2', default=0.0, type=float, help='dropout2')
parser.add_argument('--dataset_norm', default=128.0, type=float, help='dropout2')
parser.add_argument('--dataset_center', dest='dataset_center', action='store_true')
parser.add_argument('--no-dataset_center', dest='dataset_center', action='store_false')

parser.add_argument('--dense_interface', dest='dense_interface', action='store_true')
parser.add_argument('--no-dense_interface', dest='dense_interface', action='store_false')

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

parser.set_defaults(data_augmentation=True,layer_norm_res=True,layer_norm_student=True,layer_norm_2=True,skip_conn=True,last_maxpool_en=True, testmode=False,dataset_center=True, dense_interface=False)

config = parser.parse_args()
config = vars(config)
print('config  ',config)

parameters = config
TESTMODE = parameters['testmode']


lsbjob = os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob

# layer_name = parameters['layer_name']
num_feature = parameters['num_feature']
trajectory_index = parameters['trajectory_index']
sample = parameters['sample']
res = parameters['res']
trajectories_num = parameters['trajectories_num']
run_index = parameters['run_index']
dropout = parameters['dropout']
rnn_dropout = parameters['rnn_dropout']
this_run_name = parameters['run_name_prefix'] + '_j' + lsbjob + '_t' + str(int(time.time()))
parameters['this_run_name'] = this_run_name
epochs = parameters['epochs']
int_epochs = parameters['int_epochs']
student_block_size = parameters['student_block_size']
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
teacher.evaluate(trainX[45000:], trainY[45000:], verbose=2)


fe_model = teacher.layers[0]
be_model = teacher.layers[1]


#%%
#################### Get Layer features as a dataset ##########################
print('making feature data')
intermediate_layer_model = fe_model
decoder = be_model
batch_size = 32
start = 0
end = batch_size
train_data = []
validation_data = []
train_data = np.zeros([50000,res,res,num_feature])
count = 0

feature_space = 64
feature_list = 'all'

for batch in range(len(trainX)//batch_size + 1):
    count+=1
    intermediate_output = intermediate_layer_model(trainX[start:end]).numpy()
    train_data[start:end,:,:] = intermediate_output[:,:,:,:]
    start += batch_size
    end += batch_size
    

print('\nLoaded feature data from teacher')

#%%
feature_test_data = train_data[45000:]
feature_train_data = train_data[:45000]

#%%
##################### Define Student #########################################
verbose = 2
evaluate_prediction_size = 150
prediction_data_path = path +'predictions/'
shape = feature_test_data.shape
teacher_mean = np.mean(feature_test_data.reshape(shape[0]*shape[1]*shape[2], shape[3]),axis = 0)
teacher_var = np.var(feature_test_data.reshape(shape[0]*shape[1]*shape[2], shape[3]),axis = 0)
#print('teacher mean = ', teacher_mean, 'var =', teacher_var)
parameters['teacher_mean'] = teacher_mean
parameters['teacher_var'] = teacher_var
parameters['feature_list'] = feature_list   
save_model_path = path + 'saved_models/{}_feature/'.format(this_run_name)
checkpoint_filepath = save_model_path + '/{}_feature_net_ckpt'.format(this_run_name)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mean_squared_error',
    mode='min',
    save_best_only=True)
lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), 
                                               cooldown=0, 
                                               patience=5, 
                                               min_lr=0.5e-6)
early_stopper = keras.callbacks.EarlyStopping(
                                              monitor='val_mean_squared_error', 
                                              min_delta=5e-5, 
                                              patience=3, 
                                              verbose=0,
                                              mode='auto', 
                                              baseline=None, 
                                              restore_best_weights=True
                                              )


def save_model(net,path,parameters,checkpoint = True):
    home_folder = path + '{}_saved_models/'.format(this_run_name)
    if not os.path.exists(home_folder):
        os.mkdir(home_folder)
    if checkpoint:
        child_folder = home_folder + 'checkpoint/'
    else:
        child_folder = home_folder + 'end_of_run_model/'
    if not os.path.exists(child_folder):
        os.mkdir(child_folder)

    #Saving weights as numpy array
    numpy_weights_path = child_folder + '{}_numpy_weights/'.format(this_run_name)
    if not os.path.exists(numpy_weights_path):
        os.mkdir(numpy_weights_path)
    all_weights = net.get_weights()
    with open(numpy_weights_path + 'numpy_weights_{}'.format(this_run_name), 'wb') as file_pi:
        pickle.dump(all_weights, file_pi)
    #LOAD WITH - pickle.load - and load manualy to model.get_layer.set_weights()
    
    #save weights with keras
    keras_weights_path = child_folder + '{}_keras_weights/'.format(this_run_name)
    if not os.path.exists(keras_weights_path):
        os.mkdir(keras_weights_path)
    net.save_weights(keras_weights_path + 'keras_weights_{}'.format(this_run_name))
    #LOADING WITH - load_status = sequential_model.load_weights("ckpt")
    
#%%
student = student3(sample = parameters['max_length'], 
                   res = res, 
                    activation = parameters['student_nl'],
                    dropout = dropout, 
                    rnn_dropout = rnn_dropout,
                    num_feature = num_feature,
                   rnn_layer1 = parameters['rnn_layer1'],
                   rnn_layer2 = parameters['rnn_layer2'],
                   layer_norm = parameters['layer_norm_student'],
                   conv_rnn_type = parameters['conv_rnn_type'],
                   block_size = parameters['student_block_size'],
                   add_coordinates = parameters['broadcast'],
                   time_pool = parameters['time_pool'],
                   dense_interface=parameters['dense_interface'])

train_accur = []
test_accur = []
for epoch in range(epochs):
    ############################## load syclop data #################################
    print('\nReloading Syclop Data')
    
    train_dataset, test_dataset, seed_list = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = True, 
                                    add_seed = trajectories_num,
                                    trajectory_list = 0,
                                    broadcast=parameters['broadcast'],
                                    style = parameters['style'],
                                    max_length=parameters['max_length'], 
                                    noise = parameters['noise'],
                                    )
    train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample = sample)
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = sample)
    del train_dataset
    del test_dataset
    gc.collect()
    if epoch == 0:
        student.evaluate(test_dataset_x,
                    feature_test_data, verbose = 2)
    print('{}/{}'.format(epoch+1,epochs))
    student_history = student.fit(train_dataset_x,
                    feature_train_data,
                    batch_size = 32,
                    epochs = int_epochs,
                    validation_data=(test_dataset_x, feature_test_data),
                    verbose = verbose,
                    callbacks=[model_checkpoint_callback,lr_reducer,early_stopper]) #checkpoints won't really work
    if not epoch == epochs - 1:
        del train_dataset_x
        del test_dataset_x
        gc.collect()
    train_accur.append(student_history.history['mean_squared_error'])
    test_accur.append(student_history.history['val_mean_squared_error'])
train_accur = np.array(train_accur).flatten()
test_accur = np.array(test_accur).flatten()
print('{} train:'.format(student.name), train_accur)
print('{} test:'.format(student.name), test_accur)
save_model(student, save_model_path, parameters, checkpoint = False)
#student.load_weights(checkpoint_filepath) # todo!
save_model(student, save_model_path, parameters, checkpoint = True)
student.evaluate(test_dataset_x,
                feature_test_data, verbose = 2)

#%%
student_test_data = np.zeros([5000,res,res,num_feature])
student_train_data = np.zeros([45000,res,res,num_feature])
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
#Evaluate per featurefull_student_net.evaluate(test_dataset_x[0],test_dataset_y, verbose=1)
var_list = []
for feature_indx in range(num_feature):
    var = np.var(student_test_data[:,:,:,feature_indx] - feature_test_data[:,:,:,feature_indx])
    var_list.append(var)
parameters['student_var'] = var_list


with open(prediction_data_path + 'predictions_traject_{}'.format(this_run_name,), 'wb') as file_pi:
    pickle.dump((feature_test_data, student_test_data,feature_train_data,student_train_data ), file_pi) 

############################# The Student learnt the Features!! #################################################
####################### Now Let's see how good it is in classification ##########################################

#Define a Student_Decoder Network that will take the Teacher weights of the last layers:   

opt=tf.keras.optimizers.Adam(lr=1e-3)
decoder.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )



################################## Sanity Check with Teachers Features ###########################################
decoder.evaluate(feature_test_data,trainY[45000:], verbose=2)

################################## Evaluate with Student Features ###################################
print('\nEvaluating students features witout more training')
lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=1, min_lr=0.5e-6)
early_stopper = keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy', min_delta=1e-4, patience=5, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)
pre_training_accur = decoder.evaluate(student_test_data,trainY[45000:], verbose=2)
parameters['pre_training_decoder_accur'] = pre_training_accur[1]
############################ Re-train the half_net with the student training features ###########################
print('\nTraining the base newtwork with the student features')

for epoch in range(parameters['decoder_epochs']):
    print('\nRe extracting student learnt features for epoch{}'.format(epoch))
    train_dataset, test_dataset, seed_list = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True,
                                    mixed_state = True,
                                    add_seed = trajectories_num,
                                    trajectory_list = 0,
                                    broadcast=parameters['broadcast'],
                                    style = parameters['style'],
                                    max_length=parameters['max_length'],
                                    noise = parameters['noise'],
                                    )
    train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, sample = sample)
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = sample)
    for batch in range(len(train_dataset_x[0])//batch_size + 1):
        count+=1
        train_temp = student((train_dataset_x[0][start:end],train_dataset_x[1][start:end])).numpy()
        student_train_data[start:end,:,:,:] = train_temp[:,:,:,:]
        start += batch_size
        end += batch_size
    start = 0
    end = batch_size
    count = 0
    # for batch in range(len(test_dataset_x[0])//batch_size + 1):
    #     count+=1
    #     test_temp = student((test_dataset_x[0][start:end],test_dataset_x[1][start:end])).numpy()
    #     student_test_data[start:end,:,:,:] = test_temp[:,:,:,:]
    #     start += batch_size
    #     end += batch_size

    decoder_history = decoder.fit(student_train_data,
                           trainY[:45000],
                           epochs = 1  if not TESTMODE else 1,
                           batch_size = 64,
                           validation_data = (student_test_data, trainY[45000:]),
                           verbose = 2,
                           callbacks=[lr_reducer,early_stopper],)

home_folder = save_model_path + '{}_saved_models/'.format(this_run_name)
decoder.save(home_folder +'decoder_trained_model')
#%%
############################## Explore the relations between the ################################################
############################## trajectories and the accuracy of  ################################################
###################################### the test set. ############################################################
# We'll ask two questions:
    #1) For different inintialization of the test set, will we see variance in 
    #   accuracy on the test wet? 
    #2) If we record and label the correct answers for each trajectory, will
    #   we find trajectories that are better? i.e. easier tp classify? 
def full_student(student, decoder, add_coordinates = False):
    inputA = keras.layers.Input(shape=(sample, res,res,3))
    if add_coordinates:
        inputB = keras.layers.Input(shape=(sample,res,res,2))
    else:
        inputB = keras.layers.Input(shape=(sample,2))
    student_features = student((inputA,inputB))
    decoder_prediction = decoder(student_features)
    
    model = keras.models.Model(inputs=[inputA,inputB],outputs=decoder_prediction)
    
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    
    return model 

full_student_net = full_student(student, decoder,add_coordinates = parameters['broadcast'])
seed_unique = []
count_corrects = {}
var_test_accur = []
num_tests = 20
for var_test in range(num_tests):
    if trajectories_num == 0:
        break
    train_dataset, test_dataset, seed_list = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = True, 
                                    add_seed = trajectories_num,
                                    trajectory_list = 0,
                                    broadcast = parameters['broadcast'], 
                                    style = parameters['style'],
                                    max_length=parameters['max_length']
                                    )
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = sample)
    del train_dataset
    del test_dataset
    gc.collect()
    test_seed_list = np.array(seed_list[45000:])
    temp_seed_uniqe, counts = np.unique(test_seed_list, return_counts = True)
    temp_dict = dict(zip(temp_seed_uniqe, counts))
    seed_unique += [i for i in temp_seed_uniqe if i not in seed_unique]
    corrects = np.nonzero(np.argmax(full_student_net.predict(test_dataset_x),1) == trainY[45000:].reshape((-1,)))[0]
    temp_accur = len(corrects)/len(trainY[45000:])
    var_test_accur.append(temp_accur)
    correct_seeds = test_seed_list[corrects]
    #TODO this does not distinguish between the particular image and class and 
    #counts in general how many correct answers each trajectory has. We can try 
    # to also consider the images, perhaps there are some really dificults. 
    
    #Count the number of correct answers per seed
    new_temp_seed_unique, count_corrects_by_seed = np.unique(correct_seeds, return_counts = True)
    #Now let's take the percentage, divide by the number of trials (20) and add to count_corrects
    for indx, seed in enumerate(new_temp_seed_unique):
        if seed in count_corrects.keys():
            count_corrects[seed] += count_corrects_by_seed[indx]/temp_dict[seed]/num_tests
        else:
            count_corrects[seed] = count_corrects_by_seed[indx]/temp_dict[seed]/num_tests
    

    del test_dataset_x
    gc.collect()
#%%
print(count_corrects)
ymax = max(count_corrects.values())
ymin = min(count_corrects.values())
print('\nDifferent accuracies of runs')
print(var_test_accur)
parameters['count_corrects'] = count_corrects
parameters['different_test_trajectories'] = var_test_accur
plt.figure()
plt.bar(count_corrects.keys(), count_corrects.values(), width = 1 )
plt.ylim(ymin - 0.03, ymax + 0.03)
plt.title('avarage percent of correct per seed')
plt.savefig('traject_variance {}'.format(this_run_name))

############ Test the best trajectory,make it the only trajectory ############
traject_seed = np.argmax(list(count_corrects.values()))
_, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, 
                                    add_seed = 1,
                                    trajectory_list = traject_seed,
                                    broadcast = parameters['broadcast'], 
                                    style = parameters['style'],
                                    max_length=parameters['max_length']
                                    )
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset,sample = sample)
full_student_net.evaluate(test_dataset_x,trainY[45000:],verbose = 2,)
############################## Now Let's Try and Trian the student features #####################################
########################### Combining the student and the decoder and training ##################################
# print('\nTraining the student and decoder together - reinitiating the decoder before learning')
# net_weights_reinitializer(decoder)



# def full_student(student, decoder):
#     input = keras.layers.Input(shape=(sample, res,res,3))\
        
#     student_features = student(input)
#     decoder_prediction = decoder(student_features)
    
#     model = keras.models.Model(inputs=input,outputs=decoder_prediction)
    
#     opt=tf.keras.optimizers.Adam(lr=1e-3)

#     model.compile(
#         optimizer=opt,
#         loss="sparse_categorical_crossentropy",
#         metrics=["sparse_categorical_accuracy"],
#     )
    
#     return model 

# full_student_net = full_student(student, decoder)

# full_history = full_student_net.fit(train_dataset_x[0],
#                        trainY[:45000],
#                        epochs = 10  if not TESTMODE else 1,
#                        batch_size = 64,
#                        validation_data = (test_dataset_x[0], trainY[45000:]),
#                        verbose = 2,
#                        callbacks=[lr_reducer,early_stopper],)


#full_student_net.save(home_folder +'full_trained_model')   
traject_learning_dataset_update(train_accur,test_accur, decoder_history, student,parameters, name = 'full_train_103')