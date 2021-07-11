'''
run the parrallel code for N trials to get avarage - due to inability to reproduce results


out.248605

417697

100 20 8 256 1 2 0 5
out.417910

200 20 8 256 1 3 0 5
out.417965


'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
import numpy as np
import cv2
import misc
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from keras_utils import dataset_update, write_to_file, create_cifar_dataset

import tensorflow.keras as keras
import tensorflow as tf
import scipy.stats as st

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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

kernel_regularizer_list = [None, keras.regularizers.l1(),keras.regularizers.l2(),keras.regularizers.l1_l2()]
optimizer_list = [tf.keras.optimizers.Adam, tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop]
if len(sys.argv) > 1:
    paramaters = {
    'epochs' : int(sys.argv[1]),
    
    'sample' : int(sys.argv[2]),
    
    'res' : int(sys.argv[3]),
    
    'hidden_size' : int(sys.argv[4]),
    
    'concat' : int(sys.argv[5]),
    
    'regularizer' : kernel_regularizer_list[int(sys.argv[6])],
    
    'optimizer' : optimizer_list[int(sys.argv[7])],
    
    'num_trials' : int(sys.argv[8]),
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4,
    
    'run_id' : np.random.randint(1000,9000)
    }
    
else:
    paramaters = {
    'epochs' : 1,
    
    'sample' : 5,
    
    'res' : 8,
    
    'hidden_size' : 128,
    
    'concat' : 1,
    
    'regularizer' : kernel_regularizer_list[1],
    
    'optimizer' : optimizer_list[0],
    
    'num_trials' : 2,
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4,
    
    'run_id' : np.random.randint(1000,9000)
    }
   
print(paramaters)
for key,val in paramaters.items():
    exec(key + '=val')
epochs = epochs
sample = sample 
res = res 
hidden_size =hidden_size
concat = concat
regularizer = regularizer
optimizer = optimizer
num_trials = num_trials
cnn_dropout = cnn_dropout
rnn_dropout = rnn_dropout
lr = lr
run_id = run_id
n_timesteps = sample
#%%
def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def parallel_gru(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True,
                 optimizer = tf.keras.optimizers.Adam):
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

    
    ###################### CNN Chanell 1#######################################
    
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3), activation='relu',padding = 'same'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3), activation='relu',padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    
    ###################### Parallel Chanell 1##################################
    rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    if concat:
        rnn_temp = keras.layers.Concatenate()([rnn_temp,inputB])
    else:
        rnn_temp = rnn_temp
    print('flat shape after cnn1', rnn_temp.shape)
    rnn_x = keras.layers.GRU( hidden_size,input_shape=(n_timesteps, None),
                             kernel_regularizer=regularizer,
                             return_sequences=True,recurrent_dropout=2*rnn_dropout,
                             )(rnn_temp)
    print('gru hidden states 1 ', rnn_x.shape)
    ###################### CNN Chanell 2 #######################################
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2),name = 'test'),name = 'test')(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    
    ###################### Parallel Chanell 2 ##################################
    rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print('flat shape after cnn2',rnn_temp.shape)  
    if concat:
        rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp,inputB])
    else:
        rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp])
    print(' cnn2 input combined with fst hidden state', rnn_temp.shape)
    rnn_x = keras.layers.GRU( hidden_size,input_shape=(n_timesteps, None),
                             kernel_regularizer=regularizer,
                             return_sequences=True,recurrent_dropout=2*rnn_dropout,
                             )(rnn_temp)
    print('gru hidden states 2 ', rnn_x.shape)
    
    ###################### CNN Chanell 3 #######################################
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    print(x1.shape)
    
    ###################### Parallel Chanell 3 ##################################
    # rnn_temp = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    # print('flat shape after cnn3',rnn_temp.shape)
    # if concat:
    #     rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp,inputB])
    # else:
    #     rnn_temp = keras.layers.Concatenate()([rnn_x,rnn_temp])
    # print(' cnn23input combined with snd hidden state', rnn_temp.shape)
    # rnn_x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=2*rnn_dropout)(rnn_temp)
    # print('gru hidden states 3 ', rnn_x.shape)
    
    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)

    if concat:
        x = keras.layers.Concatenate()([x1,rnn_x,inputB])
    else:
        x = keras.layers.Concatenate()([x1,rnn_x])
    print(x.shape)

    # define LSTM model
    x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),
                         return_sequences=True,recurrent_dropout=rnn_dropout)(x)
    x = keras.layers.Flatten()(x)
    #add another dense, before reached 62%
    x = keras.layers.Dense(512,activation="relu")(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'parallel_gru_v1_{}'.format(concat))
    opt=optimizer(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

train_dataframe = pd.DataFrame()
test_dataframe = pd.DataFrame()
for trial in range(num_trials):
    rnn_net = parallel_gru(n_timesteps = sample, hidden_size = hidden_size,input_size = res, concat = concat)
    #keras.utils.plot_model(rnn_net, expand_nested=True,  to_file='{}.png'.format(rnn_net.name))
    #cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = res, dropout = cnn_dropout)
    
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

    print('################# {} Validation Accuracy = '.format(rnn_net.name),np.max(rnn_history.history['val_sparse_categorical_accuracy']))
    print('################# {} Training Accuracy = '.format(rnn_net.name),np.max(rnn_history.history['sparse_categorical_accuracy']))
    train_dataframe[trial] = rnn_history.history['sparse_categorical_accuracy']
    test_dataframe[trial] = rnn_history.history['val_sparse_categorical_accuracy']
    
    

train_dataframe['mean'] = train_dataframe.mean(numeric_only = True, axis = 1)
train_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(train_dataframe) - 1, loc = train_dataframe.mean(axis = 1), scale = st.sem(train_dataframe, axis = 1))[0]
train_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(train_dataframe) - 1, loc = train_dataframe.mean(axis = 1), scale = st.sem(train_dataframe, axis = 1))[1]
plt.figure()
x = np.arange(len(train_dataframe))
y = train_dataframe['mean']
plt.plot(x, y,'r',label = 'train')
plt.fill_between(x, train_dataframe['confidance-'] , train_dataframe['confidance+'],alpha = 0.4)

test_dataframe['mean'] = test_dataframe.mean(numeric_only = True, axis = 1)
test_dataframe['confidance-'] = st.t.interval(alpha = 0.95, df = len(test_dataframe) - 1, loc = test_dataframe.mean(axis = 1), scale = st.sem(test_dataframe, axis = 1))[0]
test_dataframe['confidance+'] = st.t.interval(alpha = 0.95, df = len(test_dataframe) - 1, loc = test_dataframe.mean(axis = 1), scale = st.sem(test_dataframe, axis = 1))[1]
x = np.arange(len(test_dataframe))
y = test_dataframe['mean']
plt.plot(x, y, 'r',label = 'test')
plt.fill_between(x, test_dataframe['confidance-'] , test_dataframe['confidance+'],alpha = 0.4)
plt.legend()
plt.grid()
plt.ylim(0.5,0.63)
plt.title('{} on cifar res = {} hs = {} dropout = {}, num samples = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout,sample))
plt.savefig('{} on Cifar res = {}, no upsample, val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}_{}'.format(rnn_net.name, run_id), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)

dataset_update(rnn_history, rnn_net,paramaters)    
write_to_file(rnn_history, rnn_net,paramaters)    
     