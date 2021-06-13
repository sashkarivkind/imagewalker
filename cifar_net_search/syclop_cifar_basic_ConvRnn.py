'''
The follwing code runs a test RNN with a dense layer to integrate the coordinates
network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 

Running an all convLSTM network with [32,32,64,64,128,128] dropout = 0.1 learnt nothing
ConvLSTM_True Validation Accuracy =  [0.0976, 0.1058, 0.0958, 0.1038, 0.1038, 0.1024, 0.097, 0.1024, 0.0976, 0.1058, 0.0976, 0.0986, 0.095, 0.0958, 0.097, 0.1058, 0.1038, 0.0976, 0.0958, 0.0958, 0.0976, 0.0976, 0.1064, 0.0976, 0.1038, 0.0986, 0.0986, 0.0986, 0.1024, 0.0958]
ConvLSTM_True Training Accuracy =  [0.09735555, 0.09991111, 0.10057778, 0.098177776, 0.099022225, 0.09866667, 0.09928889, 0.098844446, 0.097333334, 0.09877778, 0.099244446, 0.09762222, 0.09753333, 0.1006, 0.09808889, 0.0986, 0.10008889, 0.09957778, 0.09866667, 0.09933333, 0.098733336, 0.09802222, 0.09848889, 0.098688886, 0.10022222, 0.097666666, 0.101111114, 0.101222225, 0.10253333, 0.09933333]

The same with no dropout:
################# ConvLSTM_False Validation Accuracy =  [0.3878, 0.4404, 0.4658, 0.4984, 0.5326, 0.5346, 0.5488, 0.5436, 0.533, 0.52, 0.5176, 0.5284, 0.5044, 0.5074, 0.501, 0.4964, 0.4984, 0.5032, 0.5008, 0.502, 0.4996, 0.4982, 0.488, 0.5034, 0.505, 0.4786, 0.497, 0.514, 0.516, 0.505]
################# ConvLSTM_False Training Accuracy =  [0.29066667, 0.3886, 0.43748888, 0.47655556, 0.52477777, 0.57622224, 0.6383111, 0.71088886, 0.7854889, 0.8486222, 0.89375556, 0.9188, 0.93637776, 0.9466444, 0.95633334, 0.9564, 0.96464443, 0.9667778, 0.96691114, 0.9700222, 0.9739778, 0.9757778, 0.9756, 0.9775556, 0.9786, 0.9791333, 0.9784222, 0.9824889, 0.98224443, 0.98135555]

rnn dropout = 0, cnn dropout = 0.2, with three layer [32,64,128] out.773698
################# ConvLSTM_False Validation Accuracy =  [0.3632, 0.4382, 0.452, 0.4994, 0.5272, 0.5344, 0.5384, 0.5486, 0.5424, 0.5352, 0.5164, 0.5282, 0.5216, 0.516, 0.5014, 0.5154, 0.5212, 0.5058, 0.5052, 0.5152, 0.5112, 0.514, 0.5162, 0.5158, 0.5142, 0.4984, 0.507, 0.5144, 0.51, 0.5096]
################# ConvLSTM_False Training Accuracy =  [0.2806, 0.37866667, 0.4236889, 0.4622, 0.5073111, 0.5515778, 0.60642225, 0.6749778, 0.74655557, 0.8159556, 0.86906666, 0.89986664, 0.9198222, 0.9397111, 0.9484, 0.9533333, 0.9577111, 0.9635556, 0.96444446, 0.96793336, 0.9726, 0.97282225, 0.9732, 0.9747556, 0.9775111, 0.97746664, 0.9774889, 0.9794667, 0.98028886, 0.9795778]


rnn dropout = 0, cnn dropout = 0.2, with six layers [32,32,64,64,128,128] out.773717
################# ConvLSTM_False Validation Accuracy =  [0.3124, 0.3696, 0.3976, 0.3996, 0.4466, 0.4596, 0.4384, 0.5092, 0.4338, 0.4948, 0.4862, 0.468, 0.4668, 0.4608, 0.4498, 0.4784, 0.481, 0.4668, 0.4636, 0.4634, 0.4512, 0.4436, 0.4328, 0.4674, 0.4694, 0.4736, 0.4486, 0.4188, 0.4724, 0.4478]
################# ConvLSTM_False Training Accuracy =  [0.2573111, 0.34075555, 0.387, 0.4148, 0.44184443, 0.47144446, 0.50137776, 0.53253335, 0.5787333, 0.6349556, 0.6987778, 0.7641111, 0.81593335, 0.857, 0.8868, 0.9078444, 0.92462224, 0.9330222, 0.9415111, 0.94942224, 0.9532, 0.9556, 0.96253335, 0.9642222, 0.96493334, 0.9686222, 0.9695333, 0.9692, 0.96984446, 0.9736222]

rnn dropout = 0, cnn dropout = 0.2 with six layers [32,32,64,64,128,128] out.773719
################# ConvLSTM_False Validation Accuracy =  [0.2964, 0.3748, 0.4394, 0.4634, 0.4992, 0.5208, 0.5116, 0.5088, 0.4738, 0.5034, 0.491, 0.487, 0.4914, 0.4804, 0.478, 0.4646, 0.4802, 0.4758, 0.4716, 0.4726, 0.4798, 0.4674, 0.4732, 0.4784, 0.4906, 0.4682, 0.4868, 0.4684, 0.4878, 0.4664]
################# ConvLSTM_False Training Accuracy =  [0.24953334, 0.34342223, 0.3874889, 0.4266, 0.4738, 0.51955557, 0.5769778, 0.6451111, 0.727, 0.8041111, 0.86524445, 0.90353334, 0.92586666, 0.941, 0.9504222, 0.9550889, 0.96095556, 0.9640889, 0.9681111, 0.9703111, 0.9710444, 0.97584444, 0.9760889, 0.97513336, 0.97713333, 0.9768222, 0.98075557, 0.98142225, 0.98204446, 0.98102224]

rnn dropout = 0.1, cnn dropout = 0.2 with six layers [32,32,64,64,128,128] out.773952
################# ConvLSTM_False Validation Accuracy =  [0.2864, 0.3722, 0.3984, 0.393, 0.4652, 0.4464, 0.4842, 0.4842, 0.5006, 0.4586, 0.4768, 0.4574, 0.4714, 0.4494, 0.4574, 0.4378, 0.4618, 0.451, 0.4492, 0.4472, 0.4446, 0.4542, 0.439, 0.4416, 0.4304, 0.4468, 0.4168, 0.453, 0.4266, 0.455]
################# ConvLSTM_False Training Accuracy =  [0.23924445, 0.32406667, 0.37304443, 0.4096, 0.443, 0.4794, 0.5129111, 0.5565111, 0.6046889, 0.66695553, 0.73075557, 0.794, 0.8431778, 0.8774889, 0.9032889, 0.9224, 0.93035555, 0.9445111, 0.9454, 0.95111114, 0.95735556, 0.95724446, 0.9601333, 0.9623333, 0.9668889, 0.96882224, 0.9713111, 0.97297776, 0.9707556, 0.9740667]

rnn dropout = 0.2, cnn dropout = 0.4 with six layers [32,32,64,64,128,128] out.
################# ConvLSTM_False Validation Accuracy =  [0.258, 0.3026, 0.3396, 0.3596, 0.3954, 0.4112, 0.4476, 0.4364, 0.442, 0.4548, 0.4626, 0.4446, 0.4574, 0.4546, 0.4506, 0.4408, 0.415, 0.4194, 0.3994, 0.379, 0.3884, 0.4172, 0.3966, 0.3798, 0.4244, 0.3762, 0.4058, 0.3852, 0.3756, 0.3784]
################# ConvLSTM_False Training Accuracy =  [0.18022223, 0.2565778, 0.29637778, 0.32268888, 0.34915555, 0.37704444, 0.3999111, 0.42328888, 0.4469111, 0.4664889, 0.48937777, 0.5121111, 0.53506666, 0.5646222, 0.5948667, 0.63053334, 0.6655333, 0.6974222, 0.7324, 0.76673335, 0.79271114, 0.8174, 0.8400667, 0.8548, 0.8740889, 0.88575554, 0.8973111, 0.9061111, 0.91557777, 0.9224]

rnn dropout = 0.2, cnn dropout = 0.4 with six layers [32,32,64,64,128,128] plus another dense layer out.

'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import sys
sys.path.insert(1, "/home/orram/Documents/GitHub/imagewalker/")#'/home/labs/ahissarlab/orra/imagewalker')


import numpy as np
import cv2
import misc
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from keras_utils import *
from misc import *

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy


#Define function for low resolution lens on syclop
def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp

def bad_res102(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_AREA)
    return dwnsmp# int(sys.argv[3])

import importlib
importlib.reload(misc)
from misc import Logger
import os 


def deploy_logs():
    if not os.path.exists(hp.save_path):
        os.makedirs(hp.save_path)

    dir_success = False
    for sfx in range(1):  # todo legacy
        candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
        if not os.path.exists(candidate_path):
            hp.this_run_path = candidate_path
            os.makedirs(hp.this_run_path)
            dir_success = True# int(sys.argv[3])n_timesteps = sample, input_size = 28, input_dim = 1)
            break
    if not dir_success:
        error('run name already exists!')

    sys.stdout = Logger(hp.this_run_path+'log.log')
    print('results are in:', hp.this_run_path)
    print('description: ', hp.description)
    #print('hyper-parameters (partial):', hp.dict)

epochs = 10#int(sys.argv[1])

sample = 5#int(sys.argv[2])

res = 8#int(sys.argv[3])

hidden_size = 128#int(sys.argv[4])
   
cnn_dropout = 0.3

rnn_dropout = 0.2

n_timesteps = sample
def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def convgru(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 1, concat = False):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define LSTM model
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.1, recurrent_dropout=0.1, return_sequences=True)(inputA)
    print(x.shape)
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(x)
    # print(x.shape)
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.1, recurrent_dropout=0.1, return_sequences=True)(x)
    print(x.shape)
    x = keras.layers.Flatten()(x)
    print(x.shape)
    if concat:
        x = keras.layers.Concatenate()([x,inputB])
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'Basic_ConvLSTM_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = convgru(n_timesteps = sample, cell_size = hidden_size,input_size = 8, input_dim = 3  , concat = False)
cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = 8)

# hp = HP()
# hp.save_path = 'saved_runs'

# hp.description = "syclop cifar net search runs"
# hp.this_run_name = 'syclop_{}'.format(rnn_net.name)
# deploy_logs()
#%%
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )#bad_res_func = bad_res101, up_sample = True)

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
#%%
print("##################### Fit {} and trajectories model on training data res = {} ##################".format(cnn_net.name,res))
cnn_history = cnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 1)
print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])

#%%
print("##################### Fit {} and trajectories model on training data res = {} ##################".format(rnn_net.name,res))
rnn_history = rnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 1)

#print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])
#print('################# {} Training Accuracy = '.format(cnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
plt.legend()
plt.title('{} on cifar res = {} hs = {} dropout = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout))
plt.savefig('{} on Cifar res = {} val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict{}_{}'.format(rnn_net.name, hidden_size,cnn_dropout), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict'.format(cnn_net.name), 'wb') as file_pi:
    pickle.dump(cnn_history.history, file_pi)