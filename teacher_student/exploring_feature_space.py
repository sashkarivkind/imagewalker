#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:41:43 2021

@author: orram
"""
import os 
import pickle
import sys
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

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

layer_name = 'cnn32'#layers_names[7]#int(sys.argv[1])]
print(layer_name)
# load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
images, labels = trainX, trainY


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
testX = trainX[45000:]
testY = trainY[45000:]
#%%

path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'
#path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/'
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
else:
    model = train_model(path, trainX[:45000], trainY[:45000])

model.evaluate(trainX[45000:],trainY[45000:])

#%%
incorrects = np.nonzero(np.argmax(model.predict(trainX[45000:]),1) != trainY[45000:].reshape((-1,)))[0]
corrects = np.nonzero(np.argmax(model.predict(trainX[45000:]),1) == trainY[45000:].reshape((-1,)))[0]
# t = np.nonzero(np.argmax(model.predict(testX[50:100]),1) != testY[50:100].reshape((-1,)))
# t2 = list(incorrects[0]) + list(t[0])

#Add for loop to get all mislabels 
# incorrects = []
# for i in range()
#%%
'''
let's save the statistics of the activation of the first layer with regards to
the images seen. 
Dividing the statistics to class we get a dataframe sized 
num_classesXnum_statsXnum_examples

The type of questions I want to ask:
    1) is there a distinctable difference between the network activity of 
        images that where missclasified? PArlimenary answer - yes, it seems that
        at least with regards to the mean activity it is distinguishable between 
        true predictions and missclassified (in cnn1)
    2) is there a distinctable difference between the network activity between
        different classes? (taking only the right answer)
    3) What will be the effect of cenceling some of the kernels? cenceling the 
        ones with the strongest mean activity will cause more effect? 
    After we finish with the HR images we can ask the same with syclop and 
    compare between the representations in syclop and in HR cnn. 
    Let's explore the ideas of - 
    coherent features in cnn
    entropy in intermidiate layers of cnn
    dynaimacal feature learning
    
    '
    
'''
#for layer in model.layers:
layer_name = layer_name
exec(layer_name +' = pd.DataFrame()')
intermediate_layer_model = keras.Model(inputs = model.input,
                                       outputs = model.get_layer(layer_name).output)
feature_data_path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/feature_data/'
if os.path.exists(feature_data_path + 'train_features_{}'.format(layer_name)):
    
    #train_data = np.array(pickle.load(open('/home/orram/Documents/GitHub/imagewalker/teacher_student/train_features_{}'.format(layer_name),'rb')))
    intermediate_output = np.array(pickle.load(open(feature_data_path + 'train_features_{}'.format(layer_name),'rb')))[45000:]
else:
    intermediate_output = np.array(intermediate_layer_model(trainX[45000:]))


intermediate_corrects = intermediate_output[corrects, :, :, :]
intermediate_incorrects = intermediate_output[incorrects, :, :, :]
#%%
########################### Part I ###########################################
################### Get differences in activation between ####################
################### correct and incorrect classifications ####################
##############################################################################

mean_dataframe = pd.DataFrame()
var_dataframe = pd.DataFrame()
number_per_class = []
for i in range(10):
    
    #Get Correct statistics
    class_index = list(np.where(trainY[45000:] == i)[0])
    correct_nd_class = [idx for idx in np.arange(len(intermediate_output)) if idx in class_index and idx in corrects ]
    number_per_class.append(len(correct_nd_class))
    class_outputs = intermediate_output[correct_nd_class, : , : , :]
    shape = class_outputs.shape
    class_outputs = np.resize(class_outputs, (shape[0], shape[1]*shape[2], shape[3]))
    #################### Get intra-class statistics ###############################
    #Get the mean, std of each example - we'll focus on the first two
    #Takes in [num_images, num_features, H*W]
    #outputs [num_images, num_features]
    mean = np.mean(class_outputs,1)
    std = np.std(class_outputs,1)
    #Now get the stats of all examples of class wide
    #Takes in [num_images, num_features]
    #outputs [num_feaures]
    mean_mean = np.mean(mean, axis = 0)
    mean_std = np.std(mean, axis = 0)
    std_mean = np.mean(std, axis = 0)
    std_std = np.std(std, axis = 0)
    #Save all data from layer to dataframe 
    #We end up with a Dataframe of size (num_features), where eache location 
    #holds the mean/variance across examples of class
    mean_dataframe['mean_mean_{}'.format(i)] = mean_mean
    mean_dataframe['mean_std_{}'.format(i)] = mean_std
    var_dataframe['std_mean_{}'.format(i)] = std_mean
    var_dataframe['std_std_{}'.format(i)] = std_std
    #Get incorrect statistics
    incorrect_nd_class = [idx for idx in np.arange(len(intermediate_output)) if idx in class_index and idx in incorrects ]
    class_outputs = intermediate_output[incorrect_nd_class, : , : , :]
    shape = class_outputs.shape
    class_outputs = np.resize(class_outputs, (shape[0],shape[1]*shape[2], shape[3]))
    #################### Get intra-class statistics ###############################
    #Get the mean, std of each example - we'll focus on the first two
    #Takes in [num_images, num_features, H*W]
    #outputs [num_images, num_features]
    mean_inc = np.mean(class_outputs,1)
    std = np.std(class_outputs,1)
    #Now get the stats of all examples of class wide
    #Takes in [num_images, num_features]
    #outputs [num_feaures]
    mean_mean_inc = np.mean(mean_inc, axis = 0)
    mean_std = np.std(mean_inc, axis = 0)
    std_mean = np.mean(std, axis = 0)
    std_std = np.std(std, axis = 0)
    #Save all data from layer to dataframe 
    #We end up with a Dataframe of size (num_features), where eache location 
    #holds the mean/variance across examples of class
    mean_dataframe['mean_mean_inc_{}'.format(i)] = mean_mean_inc
    mean_dataframe['mean_std_inc_{}'.format(i)] = mean_std
    #Calculate the t value
    se = mean_dataframe['mean_std_inc_{}'.format(i)] #standart deviation NOT divided by srt(n)
    delta_mean = mean_dataframe['mean_mean_inc_{}'.format(i)] - mean_dataframe['mean_mean_{}'.format(i)]
    mean_dataframe['t_val_{}'.format(i)] = stats.ttest_ind(mean, mean_inc)[0] #TWO-TAILED
    
    var_dataframe['std_mean_inc_{}'.format(i)] = std_mean
    var_dataframe['std_std_inc_{}'.format(i)] = std_std
    
    #Shuffle data to two groups to get controll of t_val between correct and incorrect
    #do this for 5 times and take the min to compare with.
    min_t = np.ones(len(mean_mean))*5
    min_p = np.ones(len(mean_mean))*5
    max_d = np.zeros(len(mean_mean))
    for j in range(1):
        list_one = random.sample(class_index, len(correct_nd_class))
        list_two = [idx for idx in class_index if idx not in list_one]
        class_outputs = intermediate_output[list_one, : , : , :]
        shape = class_outputs.shape
        class_outputs = np.resize(class_outputs, (shape[0], shape[1]*shape[2], shape[3]))
        #################### Get intra-class statistics ###############################
        statistics = stats.describe(class_outputs, axis = 1)
        mean_cntrl = statistics[2]
        mean_stats = stats.describe(mean_cntrl, axis = 0)
        temp_mean_one = mean_stats[2]
        #now from second list:
        class_outputs = intermediate_output[list_two, : , : , :]
        shape = class_outputs.shape
        class_outputs = np.resize(class_outputs, (shape[0], shape[1]*shape[2], shape[3]))
        #################### Get intra-class statistics ###############################
        statistics = stats.describe(class_outputs, axis = 1)
        mean_cntrl_inc = statistics[2]
        mean_stats = stats.describe(mean_cntrl_inc, axis = 0)
        temp_mean_two = mean_stats[2]
        temp_mean_std = np.sqrt(mean_stats[3])
        #Calculate the t value
        se_cntrl = temp_mean_std #standart deviation NOT divided by srt(n)
        delta_mean_cntrl = temp_mean_two - temp_mean_one
        d = delta_mean_cntrl/se_cntrl
        t, p = stats.ttest_ind(mean_cntrl, mean_cntrl_inc) #TWO-TAILED
        min_t[np.where(t < min_t)] = t[np.where(t<min_t)]
        min_p[np.where(p < min_p)] = p[np.where(p<min_p)]
        max_d[np.where(d > max_d)] = d[np.where(d > max_d)]
        
    mean_dataframe['t_val_cntrl_{}'.format(i)] = min_t
    mean_dataframe['p_val_{}'.format(i)] = stats.ttest_ind(mean, mean_inc)[1] #TWO-TAILED
    mean_dataframe['p_val_cntrl_{}'.format(i)] = min_p
    mean_dataframe['differenc_p_{}'.format(i)] = stats.ttest_ind(mean, mean_inc)[1] - min_p
    mean_dataframe['delta_val_{}'.format(i)] = delta_mean/se
    mean_dataframe['delta_val_cntrl_{}'.format(i)] = d
    mean_dataframe['delta_delta_val_{}'.format(i)] = delta_mean/se - d
    test = mean_dataframe['differenc_p_{}'.format(i)][~np.isnan(mean_dataframe['differenc_p_{}'.format(i)])]
    if sum(test < 0) < len(mean_mean)//2:
        print(i, sum(test < 0))

################### Get inter-class statistics ###############################    
correct_nd_class = [idx for idx in np.arange(len(intermediate_output)) if idx in corrects ]
class_outputs = intermediate_output[correct_nd_class, : , : , :]
shape = class_outputs.shape
class_outputs = np.resize(class_outputs, (shape[0], shape[1]*shape[2], shape[3]))
#################### Get intra-class statistics ###############################
#Get the mean, std of each example - we'll focus on the first two
#Takes in [num_images, num_features, H*W]
#outputs [num_images, num_features]
mean = np.mean(class_outputs,1)
std = np.std(class_outputs,1)
#Now get the stats of all examples of class wide
#Takes in [num_images, num_features]
#outputs [num_feaures]
mean_mean = np.mean(mean, axis = 0)
mean_std = np.std(mean, axis = 0)
std_mean = np.mean(std, axis = 0)
std_std = np.std(std, axis = 0)
#Save all data from layer to dataframe 
#We end up with a Dataframe of size (num_features), where eache location 
#holds the mean/variance across examples of class
mean_dataframe['mean_mean_all'] = mean_mean
mean_dataframe['mean_std_all'] = mean_std
var_dataframe['std_mean_all'] = std_mean
var_dataframe['std_std_all'] = std_std
#Get incorrect statistics
incorrect_nd_class = [idx for idx in np.arange(len(intermediate_output)) if idx in incorrects ]
class_outputs = intermediate_output[incorrect_nd_class, : , : , :]
shape = class_outputs.shape
class_outputs = np.resize(class_outputs, (shape[0], shape[1]*shape[2], shape[3]))
#################### Get intra-class statistics ###############################
#Get the mean, variance, skewness, kurtosis of each example - we'll focus on the first two
#Takes in [num_images, num_features, H*W]
#outputs [num_images, num_features]
mean_inc = np.mean(class_outputs,1)
std = np.std(class_outputs,1)
#Now get the stats of all examples of class wide
#Takes in [num_images, num_features]
#outputs [num_feaures]
mean_mean_inc = np.mean(mean_inc, axis = 0)
mean_std = np.std(mean_inc, axis = 0)
std_mean = np.mean(std, axis = 0)
std_std = np.std(std, axis = 0)
#Save all data from layer to dataframe 
#We end up with a Dataframe of size (num_features), where eache location 
#holds the mean/variance across examples of class
mean_dataframe['mean_mean_inc_all'] = mean_mean_inc
mean_dataframe['mean_std_inc_all'] = mean_std
#Calculate the t value
se = mean_dataframe['mean_std_all']
delta_mean = mean_dataframe['mean_mean_all'] - mean_dataframe['mean_mean_inc_all']
mean_dataframe['t_val_{}'.format(i)] = stats.ttest_ind(mean, mean_inc)[0]
var_dataframe['std_mean_inc_all'] = std_mean
var_dataframe['std_std_inc_all'] = std_std
#Shuffle data to two groups to get controll of t_val between correct and incorrect
#do this for 5 times and take the min to compare with.
min_t = np.ones(len(mean_mean))*5
min_p = np.ones(len(mean_mean))*5
max_d = np.zeros(len(mean_mean))
for j in range(5):
    list_one = random.sample(list(np.arange(len(intermediate_output))), len(correct_nd_class))
    list_two = [idx for idx in class_index if idx not in list_one]
    class_outputs = intermediate_output[list_one, : , : , :]
    shape = class_outputs.shape
    class_outputs = np.resize(class_outputs, (shape[0], shape[1]*shape[2], shape[3]))
    #################### Get intra-class statistics ###############################
    statistics = stats.describe(class_outputs, axis = 1)
    mean_cntrl = statistics[2]
    mean_stats = stats.describe(mean_cntrl, axis = 0)
    temp_mean_one = mean_stats[2]
    #now from second list:
    class_outputs = intermediate_output[list_two, : , : , :]
    shape = class_outputs.shape
    class_outputs = np.resize(class_outputs, (shape[0], shape[1]*shape[2], shape[3]))
    #################### Get intra-class statistics ###############################
    statistics = stats.describe(class_outputs, axis = 1)
    mean_cntrl_inc = statistics[2]
    mean_stats = stats.describe(mean_cntrl_inc, axis = 0)
    temp_mean_two = mean_stats[2]
    temp_mean_std = np.sqrt(mean_stats[3])
    #Calculate the t value
    se_cntrl = temp_mean_std
    delta_mean_cntrl = temp_mean_two - temp_mean_one
    d = delta_mean_cntrl/se_cntrl
    t, p = stats.ttest_ind(mean_cntrl, mean_cntrl_inc) #TWO-TAILED
    min_t[np.where(t < min_t)] = t[np.where(t<min_t)]
    min_p[np.where(p < min_p)] = p[np.where(p<min_p)]
    max_d[np.where(d > max_d)] = d[np.where(d>max_d)]
    
mean_dataframe['t_val_cntrl_all'] = min_t
mean_dataframe['p_val_all'] = stats.ttest_ind(mean, mean_inc)[1]
mean_dataframe['p_val_cntrl_all'] = min_p
mean_dataframe['differenc_p_all'] = stats.ttest_ind(mean, mean_inc)[1] - min_p
mean_dataframe['delta_val_all'] = delta_mean/se
mean_dataframe['delta_val_cntrl_all'] = d
mean_dataframe['delta_delta_val_all'] = delta_mean/se - d
test = mean_dataframe['differenc_p_all'][~np.isnan(mean_dataframe['differenc_p_all'])]
if sum(test < 0) < len(mean_mean)//2:
    print('all', sum(test < 0))
#%%
########################### Part II ###########################################
################### Get differences in activation between ####################
###################           different classes           ####################
##############################################################################

#Let's create a matrix/heatmap who doesn't like one of these??
#num_classXnum_classXnum_features

p_matrix = np.zeros([11,11,shape[3]])
d_matrix = np.zeros([11,11,shape[3]])
for i in range(11):
    for j in range(11):
        if i == 10:
            sm = mean_dataframe['mean_mean_all']
            sv = mean_dataframe['mean_std_all']
            n = len(intermediate_output)
        else:
            sm = mean_dataframe['mean_mean_{}'.format(i)]
            sv = mean_dataframe['mean_std_{}'.format(i)]
            n = number_per_class[i]
        if j == 10:
            m = mean_dataframe['mean_mean_all']
        else:
            m = mean_dataframe['mean_mean_{}'.format(j)]

        tt = (sm-m)/np.sqrt(sv/float(n))  # t-statistic for mean
        delta = (sm-m)/sv  # dif in std values (no normalization by n)
        pval = stats.t.sf(np.abs(tt), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
        p_matrix[i, j, :] = pval
        d_matrix[i, j, :] = delta
#%%
########################### Part III ###########################################
################### What happens when we cancel some      ####################
###################          of the features???           ####################
##############################################################################

#We'll cancel (set kernel weights to zero) the kernal with maximum activity 
#of each class, min activity, plus 5 random kernels and record the overall 
#effect and the micro level class effect. 
def evaluate_and_confusion(model, testx = testX, testy = testY, 
                           batch_size = 64, number_per_class = number_per_class):
    '''
    generate a prediction given the current model and compare the number of misslabels per class. 

    '''
    #place holder to store all the correct answers per class
    true_labels = np.zeros(10)
    start = 0
    end = batch_size
    for i in range(len(testx)//batch_size + 1):
        data = testx[start:end]
        labels = testy[start:end]
        prediction = model.predict(data)
        correct_labels = np.nonzero(np.argmax(prediction,1) == labels.reshape((-1,)))[0]
        for j in range(10):
            class_index = list(np.where(labels == j)[0])
            correct_class = len([l for l in correct_labels if l in class_index])
            true_labels[j] += correct_class
        start += batch_size
        end += batch_size
    correct_delta = true_labels - number_per_class #Get the delta between how much
                                                    #was labeled before and now
    delta_percentage = true_labels/number_per_class
    accuracy = sum(true_labels)/len(testy) 
    
    return accuracy, correct_delta, delta_percentage
    
#accuracy , coorect_delta = evaluate_and_confusion(model = model,testx = testX, testy = testY, batch_size = 64, number_per_class = number_per_class)
# model_copy = keras.models.clone_model(model)
# accuracy , coorect_delta, delta_percentage = evaluate_and_confusion(model = model,testx = testX, testy = testY, batch_size = 64, number_per_class = number_per_class)
#%% 
model_copy = keras.models.clone_model(model)
model_copy.set_weights(model.get_weights()) 
model.evaluate(trainX[45000:],trainY[45000:])
opt=tf.keras.optimizers.Adam(lr=1e-3)
feature_importance_metrix = np.zeros([11, 7, 11])
feature_percentage_metrix = np.zeros([11, 7, 10])

def set_weights_to_zero(model_copy, orig_model, layer_name, index):
    weights = np.array(orig_model.get_layer(layer_name).weights[0])
    #print(weights.shape)
    #zero out the desired kernel - the weights are in dimention:
        #[kernel_size, kernel_size, prev_dim, next_dim]
    weights[:,:,:, index] = 0
    new_weights = [weights, np.array(orig_model.get_layer(layer_name).weights[1])]
    model_copy.get_layer(layer_name).set_weights(new_weights)
    return model_copy
#%%
index_list = []
for i in range(11):
    corrent_indexs = []
    #get most active kernel/feature
    if i == 10:
        mean_activity = mean_dataframe['mean_mean_all']
    else:
        mean_activity = np.array(mean_dataframe['mean_mean_{}'.format(i)])
    
    ########################### Max Activity #################################
    sorted_index = mean_activity.argsort()
    max_activity_index = np.argmax(mean_activity)
    if max_activity_index in np.array(index_list):
        max_activity_index = list(sorted_index)[-2]
    corrent_indexs.append(max_activity_index)
    #WHAT IS THE DIFFERENCE WITH OTHER MEAN ACTIVITY VALUES????
    model_copy = set_weights_to_zero(model_copy = model_copy, orig_model = model, 
                  layer_name = layer_name, index = max_activity_index)

    accuracy , coorect_delta, delta_percentage = evaluate_and_confusion(model = model_copy)
    feature_importance_metrix[i, 0, :] = np.append(accuracy,coorect_delta)
    feature_percentage_metrix[i, 0, :] = delta_percentage
    ########################## Min Activity ##################################
    min_activity_index = np.argmin(mean_activity)
    min_count = 0
    while mean_activity[min_activity_index] == 0 :
        min_activity_index = sorted_index[min_count +1]
        min_count += 1
        if min_count > 20:
            print(min_count, mean_activity[min_activity_index])
    if min_activity_index in np.array(index_list):
        min_activity_index = sorted_index[min_count + 1]
    corrent_indexs.append(min_activity_index)
    model_copy = set_weights_to_zero(model_copy = model_copy, orig_model = model, 
                  layer_name = layer_name, index = min_activity_index)

    accuracy , coorect_delta, delta_percentage = evaluate_and_confusion(model = model_copy)
    feature_importance_metrix[i, 1, :] = np.append(accuracy,coorect_delta)
    feature_percentage_metrix[i, 1, :] = delta_percentage
    ########################## Random Features ###############################
    
    for w in range(5):
        count = 0
        index = np.random.randint(0, len(mean_dataframe['mean_mean_all']))
        while index in corrent_indexs:
            index = np.random.randint(0, len(mean_dataframe['mean_mean_all']))
            count+=1
            if count > 10:
                break
        corrent_indexs.append(index)
        model_copy = set_weights_to_zero(model_copy = model_copy, orig_model = model, 
                      layer_name = layer_name, index = index)
    
        accuracy , coorect_delta, delta_percentage = evaluate_and_confusion(model = model_copy)
        feature_importance_metrix[i, w + 2, :] = np.append(accuracy,coorect_delta)
        feature_percentage_metrix[i, w + 2, :] = delta_percentage
    index_list.append(corrent_indexs)

#%%
############################ SAVE EVERYTHING #################################
mean_dataframe.to_pickle(path + 'saved_dataframes/mean_dataframe_{}'.format(layer_name))
all_matrices = {'p_matrix': p_matrix,
                'd_matrix': d_matrix,
                'feature_importance_metrix': feature_importance_metrix,
                'feature_percentage_metrix': feature_percentage_metrix}
with open(path + 'saved_dataframes/matrices_{}'.format(layer_name), 'wb') as f:
    pickle.dump(all_matrices, f)


#%%
#Zero out layers to see importance
silancing_features_dataframe = pd.DataFrame()

testY = trainY[45000:]
for num_to_zero in np.arange(1, len(mean_dataframe), 10):
    vals_to_save = {}
    vals_to_save['num_to_zero'] = num_to_zero
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights()) 
    mean_activity = mean_dataframe['mean_mean_all']
    sorted_index = mean_activity.argsort()
    max_index = list(sorted_index)[-num_to_zero:]
    for indx in max_index:
        model_copy = set_weights_to_zero(model_copy = model_copy, orig_model = model_copy, 
                          layer_name = layer_name, index = indx)
    
    accuracy , coorect_delta, delta_percentage = evaluate_and_confusion(model = model_copy,testx = testX, testy = testY, batch_size = 64, number_per_class = number_per_class)
    vals_to_save['strongest'] = accuracy
    
    random_silance = []
    for trial in range(2):
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        count = 0
        for indx in np.random.randint(0,128,num_to_zero):
            count += 1
            model_copy = set_weights_to_zero(model_copy = model_copy, orig_model = model_copy, 
                          layer_name = layer_name, index = indx)
    
        accuracy , coorect_delta, delta_percentage = evaluate_and_confusion(model = model_copy,testx = testX, testy = testY, batch_size = 64, number_per_class = number_per_class)
        random_silance.append(accuracy)
    
    vals_to_save['random'] = np.mean(random_silance)
    
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights()) 
    mean_activity = mean_dataframe['mean_mean_all']
    sorted_index = mean_activity.argsort()
    min_index = list(sorted_index)[:num_to_zero]
    for indx in min_index:
        model_copy = set_weights_to_zero(model_copy = model_copy, orig_model = model_copy, 
                          layer_name = layer_name, index = indx)
    
    accuracy , coorect_delta, delta_percentage = evaluate_and_confusion(model = model_copy,testx = testX, testy = testY, batch_size = 64, number_per_class = number_per_class)
    vals_to_save['weakest'] = accuracy
    
    silancing_features_dataframe = silancing_features_dataframe.append(vals_to_save, ignore_index = True)

silancing_features_dataframe.to_pickle(path + 'saved_dataframes/silancing_features_{}'.format(layer_name))