'''
The follwing code runs a test lstm network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 

cnn_dropout = 0.4 rnn_dropout = 0.2 , WITH cnivlstm_dropout samples = 10, h = 256, epochs = 150, convlstm activation = relu - out.309660 (Based on best results from cnn - gru)
very BAD results - 
################# convlstm_cnn_mix_v0_True Validation Accuracy =  [0.28940001130104065, 0.3102000057697296, 0.2964000105857849, 0.31040000915527344, 0.29980000853538513, 0.33739998936653137, 0.3325999975204468, 0.3677999973297119, 0.362199991941452, 0.34540000557899475, 0.3203999996185303, 0.311599999666214, 0.36160001158714294, 0.34139999747276306, 0.3720000088214874, 0.38920000195503235, 0.3675999939441681, 0.3741999864578247, 0.36320000886917114, 0.39320001006126404, 0.38679999113082886, 0.3686000108718872, 0.3540000021457672, 0.3919999897480011, 0.40639999508857727, 0.3959999978542328, 0.38960000872612, 0.4000000059604645, 0.37959998846054077, 0.375, 0.3882000148296356, 0.37700000405311584, 0.37599998712539673, 0.38999998569488525, 0.3837999999523163, 0.3747999966144562, 0.387800008058548, 0.37059998512268066, 0.4156000018119812, 0.39160001277923584, 0.3869999945163727, 0.3959999978542328, 0.39719998836517334, 0.4075999855995178, 0.41019999980926514, 0.35839998722076416, 0.39340001344680786, 0.4018000066280365, 0.3903999924659729, 0.39800000190734863]
################# convlstm_cnn_mix_v0_True Training Accuracy =  [0.2208888828754425, 0.2883777916431427, 0.31922221183776855, 0.33962222933769226, 0.3580666780471802, 0.36942222714424133, 0.3785777688026428, 0.3898666799068451, 0.3991999924182892, 0.40415555238723755, 0.40951111912727356, 0.4152222275733948, 0.4224666655063629, 0.4275111258029938, 0.4343999922275543, 0.4385777711868286, 0.43993332982063293, 0.44620001316070557, 0.44922223687171936, 0.450688898563385, 0.4569999873638153, 0.45651111006736755, 0.45597776770591736, 0.4597555696964264, 0.46560001373291016, 0.46533334255218506, 0.4652888774871826, 0.47055554389953613, 0.472555547952652, 0.4762444496154785, 0.4754444360733032, 0.4771333336830139, 0.47886666655540466, 0.4812222123146057, 0.48080000281333923, 0.4838666617870331, 0.48562222719192505, 0.48624444007873535, 0.4858444333076477, 0.4900444447994232, 0.48955556750297546, 0.49133333563804626, 0.49051111936569214, 0.4947555661201477, 0.49442222714424133, 0.4937777817249298, 0.49593332409858704, 0.4984889030456543, 0.49720001220703125, 0.5015555620193481]

cnn_dropout = 0.4 rnn_dropout = 0.2 , WITH cnivlstm_dropout samples = 10, h = 256, epochs = 150, convlstm activation = relu - out.309660 (Based on best results from cnn - gru)
Taking out the first cnn layer and leaving only the convlstm out.372499



'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
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
    return dwnsmp

# import importlib
# importlib.reload(misc)
# from misc import Logger
# import os 


# def deploy_logs():
#     if not os.path.exists(hp.save_path):
#         os.makedirs(hp.save_path)

#     dir_success = False
#     for sfx in range(1):  # todo legacy
#         candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
#         if not os.path.exists(candidate_path):
#             hp.this_run_path = candidate_path
#             os.makedirs(hp.this_run_path)
#             dir_success = Truecnn_net = cnn_one_img(n_timesteps = sample, input_size = 28, input_dim = 1)
#             break
#     if not dir_success:
#         error('run name already exists!')

#     sys.stdout = Logger(hp.this_run_path+'log.log')
#     print('results are in:', hp.this_run_path)
#     print('description: ', hp.description)
#     #print('hyper-parameters (partial):', hp.dict)
if len(sys.argv) > 1:
    paramaters = {
    'epochs' : int(sys.argv[1]),
    
    'sample' : int(sys.argv[2]),
    
    'res' : int(sys.argv[3]),
    
    'hidden_size' : int(sys.argv[4]),
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4
    }
    
else:
    paramaters = {
    'epochs' : 1,
    
    'sample' : 5,
    
    'res' : 8,
    
    'hidden_size' : 128,
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4
    }
    
print(paramaters)
for key,val in paramaters.items():
    exec(key + '=val')
n_timesteps = sample
def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def convlstm(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True):
    '''
    
    CNN RNN combination that extends the CNN to a network that achieves 
    ~80% accuracy on full res cifar.

    Parameters
    ----------
    n_timesteps : TYPE, optional
        DESCRIPTION. The default is 5.
    img_dim : TYPE, optional
        DESCRIPTION. The default is 32.
    hidden_size : TYPE, optional
        DESCRIPTION. The default is 128.
    input_size : TYPE, optional
        DESCRIPTION. The default is 32.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    '''
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model
    
    x1=keras.layers.ConvLSTM2D(32,(3,3), padding = 'same', dropout = cnn_dropout, recurrent_dropout=rnn_dropout,return_sequences=True)(inputA)
    #x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3), activation='relu',padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    print(x1.shape)


    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)

    # define LSTM model
    x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=rnn_dropout)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'convlstm_cnn_mix_v0_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = convlstm(n_timesteps = sample, hidden_size = hidden_size,input_size = res, concat = True)
#keras.utils.plot_model(rnn_net, expand_nested=True,  to_file='{}.png'.format(rnn_net.name))
#cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = res, dropout = cnn_dropout)


# hp = HP()
# hp.save_path = 'saved_runs'

# hp.description = "syclop cifar net search runs"
# hp.this_run_name = 'syclop_{}'.format(rnn_net.name)
# deploy_logs()

train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )
                                    #bad_res_func = bad_res101, up_sample = True)

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)


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
    verbose = 0)

# print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])
# print('################# {} Training Accuracy = '.format(cnn_net.name),rnn_history.history['sparse_categorical_accuracy'])

print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
# plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
# plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
plt.legend()
plt.title('{} on cifar res = {} hs = {} dropout = {}, num samples = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout,sample))
plt.savefig('{} on Cifar res = {}, no upsample, val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict{}_{}'.format(rnn_net.name, hidden_size,cnn_dropout), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
# with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict'.format(cnn_net.name), 'wb') as file_pi:
#     pickle.dump(cnn_history.history, file_pi)
    