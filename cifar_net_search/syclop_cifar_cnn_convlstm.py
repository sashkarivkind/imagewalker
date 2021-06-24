'''
The follwing code runs a test lstm network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 


epochs = 50, sample = 10, h = 256, lr = 1e-3 out.419516
################# cnn_convlstm_True Validation Accuracy =  [0.3677999973297119, 0.41780000925064087, 0.42480000853538513, 0.46799999475479126, 0.4830000102519989, 0.47999998927116394, 0.517799973487854, 0.5242000222206116, 0.5135999917984009, 0.545799970626831, 0.5527999997138977, 0.5490000247955322, 0.5612000226974487, 0.5450000166893005, 0.5392000079154968, 0.5436000227928162, 0.5440000295639038, 0.5509999990463257, 0.5712000131607056, 0.5666000247001648, 0.5655999779701233, 0.5641999840736389, 0.5565999746322632, 0.5756000280380249, 0.5702000260353088, 0.5641999840736389, 0.5802000164985657, 0.5573999881744385, 0.5852000117301941, 0.5766000151634216, 0.5817999839782715, 0.5845999717712402, 0.5839999914169312, 0.5866000056266785, 0.5558000206947327, 0.5902000069618225, 0.5726000070571899, 0.579800009727478, 0.5831999778747559, 0.5835999846458435, 0.5690000057220459, 0.5853999853134155, 0.5917999744415283, 0.5834000110626221, 0.590399980545044, 0.5817999839782715, 0.5821999907493591, 0.5871999859809875, 0.5934000015258789, 0.590399980545044]
################# cnn_convlstm_True Training Accuracy =  [0.2563999891281128, 0.37975555658340454, 0.4263555407524109, 0.45080000162124634, 0.4720666706562042, 0.48091110587120056, 0.4981333315372467, 0.505577802658081, 0.515666663646698, 0.5201555490493774, 0.5304222106933594, 0.5345777869224548, 0.5436221957206726, 0.5472221970558167, 0.5491999983787537, 0.5560888648033142, 0.5580666661262512, 0.5659555792808533, 0.5663555264472961, 0.571399986743927, 0.5729110836982727, 0.5763555765151978, 0.5796666741371155, 0.5829777717590332, 0.5868444442749023, 0.5905333161354065, 0.5912222266197205, 0.59297776222229, 0.5947333574295044, 0.5962888598442078, 0.600777804851532, 0.6038888692855835, 0.6055777668952942, 0.6070888638496399, 0.6077555418014526, 0.6114444732666016, 0.6135333180427551, 0.6132222414016724, 0.6160666942596436, 0.6183333396911621, 0.6185333132743835, 0.6252222061157227, 0.6219333410263062, 0.6187777519226074, 0.6248666644096375, 0.6288889050483704, 0.6315555572509766, 0.6317999958992004, 0.6333777904510498, 0.6329555511474609]

epochs = 50, sample = 5, h = 128, lr = 1e-3 out.425499
################# cnn_convlstm_True Validation Accuracy =  [0.311599999666214, 0.39660000801086426, 0.41499999165534973, 0.4569999873638153, 0.4742000102996826, 0.48260000348091125, 0.5073999762535095, 0.5013999938964844, 0.4862000048160553, 0.5192000269889832, 0.527999997138977, 0.5281999707221985, 0.5267999768257141, 0.5335999727249146, 0.5317999720573425, 0.5450000166893005, 0.551800012588501, 0.5414000153541565, 0.5609999895095825, 0.5551999807357788, 0.5437999963760376, 0.5591999888420105, 0.5508000254631042, 0.5577999949455261, 0.5591999888420105, 0.5680000185966492, 0.5586000084877014, 0.5669999718666077, 0.5734000205993652, 0.5735999941825867, 0.5640000104904175, 0.5785999894142151, 0.5669999718666077, 0.5788000226020813, 0.5753999948501587, 0.5831999778747559, 0.5681999921798706, 0.5669999718666077, 0.5788000226020813, 0.5849999785423279, 0.5770000219345093, 0.5699999928474426, 0.5753999948501587, 0.5834000110626221, 0.5756000280380249, 0.574999988079071, 0.5827999711036682, 0.5720000267028809, 0.5899999737739563, 0.579200029373169]
################# cnn_convlstm_True Training Accuracy =  [0.23742222785949707, 0.3535333275794983, 0.39693334698677063, 0.41724443435668945, 0.4415111243724823, 0.4539111256599426, 0.4665111005306244, 0.4772222340106964, 0.48704445362091064, 0.4934000074863434, 0.49951112270355225, 0.5089555382728577, 0.5167555809020996, 0.5209333300590515, 0.5235999822616577, 0.5284444689750671, 0.5305111408233643, 0.5354889035224915, 0.5397555828094482, 0.5409777760505676, 0.5472221970558167, 0.5471333265304565, 0.555400013923645, 0.5550888776779175, 0.5568666458129883, 0.5606889128684998, 0.5614222288131714, 0.5644222497940063, 0.5658666491508484, 0.5699777603149414, 0.5725555419921875, 0.5740444660186768, 0.5759555697441101, 0.573711097240448, 0.5773333311080933, 0.5770888924598694, 0.5797333121299744, 0.5830444693565369, 0.5852888822555542, 0.5862444639205933, 0.5860221982002258, 0.5873110890388489, 0.5882444381713867, 0.5902888774871826, 0.5925777554512024, 0.5944888591766357, 0.5947111248970032, 0.5948222279548645, 0.5959333181381226, 0.5995333194732666]


epochs = 200, sample = 10, h = 256, lr = 1e-3 out.438761
epochs = 200, sample = 5, h = 128, lr = 1e-3 out.439326

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
    return dwnsmp

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

cnn_dropout = 0.4

rnn_dropout = 0.2

lr = 5e-4

def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def cnn_convlstm(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True):
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

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    #x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)
    print(x1.shape)

    #x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    #print(x1.shape)

    print(x1.shape)

    # define LSTM model
    x1 = keras.layers.ConvLSTM2D(64, 3, padding = 'same',return_sequences=True,dropout = cnn_dropout, recurrent_dropout=rnn_dropout)(x1)
    x1 = keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape, inputB.shape)
    if concat:
        x =keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    x = keras.layers.Flatten()(x)
    print(x.shape)
    x = keras.layers.Dense(256,activation="relu")(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    print(x.shape)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'cnn_convlstm_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = cnn_convlstm(n_timesteps = sample, hidden_size = hidden_size,input_size = 8, concat = True)
#cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = 32)
#%%
# hp = HP()
# hp.save_path = 'saved_runs'

# hp.description = "syclop cifar net search runs"
# hp.this_run_name = 'syclop_{}'.format(rnn_net.name)
# deploy_logs()

train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0)
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


print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
plt.legend()
plt.title('{} on cifar res = {} hs = {} dropout = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout))
plt.savefig('{} on Cifar res = {} val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict{}_{}'.format(rnn_net.name, hidden_size,cnn_dropout), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
    