#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:41:43 2021

@author: orram
"""
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
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import time
import pickle
import argparse
from feature_learning_utils import  load_student, student3
from sklearn.decomposition import PCA

from keras_utils import create_trajectory
from dataset_utils import Syclopic_dataset_generator, test_num_of_trajectories
from vanvalenlab_convolutional_recurrent import ConvGRU2D
print(os.getcwd() + '/')

parser = argparse.ArgumentParser()

#general parameters
parser.add_argument('--path', default=os.getcwd()+'/', help = 'the path from where to take the student and save the data')
parser.add_argument('--run_name', default = 'saved_models/noname_j983145_t1637233976_feature/noname_j983145_t1637233976_saved_models/fro_student_and_decoder_trained')

parser.add_argument('--test_mode', default=0, type = int, help = 'if True will run over 10 features to test')
parser.add_argument('--train_decoder', default=1, type = int, help = 'RETRAIN DECODER - to turn off if we fix the saving issue')
parser.add_argument('--num_of_runs', default = 5, type = int, help = 'How many times to run the generator over a single feature to get distrebution of activation values')
parser.add_argument('--feature_silencing_tech', default = 'random', help = 'What techniuqe to use in order to silance out a feature - options: zero, shuffle, random')
parser.add_argument('--loading_style', default = 'direct')
parser.add_argument('--feature_importance', default=1, type=int, help='to run feature importance test or not.')
parser.add_argument('--gradient_ascent_iterations', default=10, type = int)
parser.add_argument('--loud', default = 0, type = int)
parser.add_argument('--find_AM', default = 0, type = int)
parser.add_argument('--AM_list_path', default = 'temp_loder', help = 'path to the saved data from the run to use for the silancing part if --find_AM == 0')
parser.add_argument('--one_by_one_slice', default=0, type = int , help = 'weather to check every single feature accuracy signifacent')
parser.add_argument('--num_random_evaluations',default=5, type = int, help = 'How many repetitions of random drew to do in silancing analysis')
parser.add_argument('--testbad', default=1,type = int)
parser.add_argument('--noise', default=0.05, type = float, help = 'Added noise to generated image during learning')
parser.add_argument('--regularization', default = 'batchnorm', type = str, help = 'What regularization to use in the generators')
parser.add_argument('--t_lim', default=0.15, type = float, help = 'set limit on temporal value to use to isolate spacial AM importance')
parser.add_argument('--s_lim', default=0.15, type = float, help = 'set limit on spacial value to use to isolate temporal AM importance')
parser.add_argument('--outlier', default = 0, type = int)
parser.add_argument('--student_AM_layer', default = 'convLSTM31', type = str, help ='Choose the layer of the student to run AM search over')
config = parser.parse_args()
config = vars(config)
if config['test_mode']:
    config['gradient_ascent_iterations'] = 1
    config['num_of_runs'] = 2
    config['loud'] = 0
print('config  ',config)
#%%
############################### Get Trained Student ##########################3
if config['loading_style'] == 'direct':
    one = False
    for indx, i in enumerate(config['run_name']):
        if i == 'n':
            if not one:
                one = True
                start_indx = indx
        if one:
            if i == '/':
                break
    run_name = config['run_name'][start_indx:indx]
else:
    run_name = config['run_name']
num_samples = 0
lsbjob = os.getenv('LSB_JOBID')
lsbjob = '_' if lsbjob is None else lsbjob
dump_folder = config['path'] + run_name + '_' + lsbjob #.format(time.strftime("%m-%d-%H_%M_%S"))
config['dump_folder'] = dump_folder
os.mkdir(dump_folder)
if config['loading_style'] == 'old':
    student, parameters, decoder = load_student(path = config['path'], 
                                           run_name = run_name, 
                                           num_samples = num_samples,
                                           student=None)
elif config['loading_style'] == 'direct':
    student_and_decoder = tf.keras.models.load_model(config['run_name'],custom_objects={'ConvGRU2D':ConvGRU2D})
    student = student_and_decoder.get_layer(name = 'student_3')
    decoder = student_and_decoder.get_layer(name = 'backend')
    # _, parameters, _ = load_student(path = config['path'], 
    #                                        run_name = 'noname_j96959_t1636556904', 
    #                                        num_samples = num_samples,
    #                                        student=None)
    parameters = {'run_name_prefix': 'noname', 'run_index': 10, 'verbose': 2, 'n_classes': 10, 'testmode': False, 'epochs': 100, 'int_epochs': 1, 'decoder_epochs': 40, 'num_feature': 64, 'rnn_layer1': 64, 'rnn_layer2': 128, 'time_pool': 'average_pool', 'student_block_size': 2, 'upsample': 0, 'conv_rnn_type': 'gru', 'student_nl': 'elu', 'dropout': 0.0, 'rnn_dropout': 0.0, 'pretrained_student_path': None, 'pos_det': None, 'decoder_optimizer': 'Adam', 'skip_student_training': False, 'fine_tune_student': False, 'layer_norm_student': True, 'batch_norm_student': False, 'val_set_mult': 5, 'trajectory_index': 0, 'n_samples': 10, 'res': 8, 'trajectories_num': -1, 'broadcast': 0, 'style': 'spiral_2dir2', 'loss': 'mean_squared_error', 'noise': 0.5, 'max_length': 10, 'random_n_samples': 0, 'teacher_net': '/home/labs/ahissarlab/arivkind/imagewalker/keras-resnet/model_52191__1631198121.hdf', 'resblocks': 3, 'student_version': 3, 'last_layer_size': 128, 'dropout1': 0.2, 'dropout2': 0.0, 'dataset_norm': 128.0, 'syclopic_norm': 1.0, 'dataset_center': True, 'dense_interface': False, 'layer_norm_res': True, 'layer_norm_2': True, 'skip_conn': True, 'last_maxpool_en': True, 'resnet_mode': False, 'nl': 'relu', 'stopping_patience': 10, 'learning_patience': 5, 'manual_suffix': '', 'shuffle_traj': False, 'data_augmentation': True, 'rotation_range': 0.0, 'width_shift_range': 0.1, 'height_shift_range': 0.1, 'time_sec': 0.3, 'traj_out_scale': 4.0, 'snellen': True, 'vm_kappa': 0.0, 'this_run_name': 'noname_j983145_t1637233976'}
student.summary()
config['parameters'] = parameters
config['folder_name'] = dump_folder
run_name = run_name + '_{}'.format(time.strftime("%m_%d_%H_%M"))
print('config  ',config)
#%%
student.summary()
# The dimensions of our input image
img_width = 8
img_height = 8
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.

#%%

# Set up a model that returns the activation values for our target layer
# Looking on the features of the output, what the student learns to imitate
feature_extractor = student

# Set up a model that returns the activation values for our target layer
layer = student.get_layer(name = config['student_AM_layer'])
feature_extractor = keras.Model(inputs=student.inputs, outputs=layer.output)
feature_extractor.trainable = False
#%%
def compute_loss(filter_index,input_image):
    traject = np.array(create_trajectory((32,32),sample = 10, style = 'spiral'))#tf.random.uniform((10,2))
    traject = np.expand_dims(traject, 0)
   
    input_image += tf.random.normal(input_image.shape, stddev = config['noise'])
    activation = feature_extractor((input_image,traject))
    # We avoid border artifacts by only involving non-border pixels in the loss.
    if config['loud']:
        print('input_image', input_image.shape)#
        print('feature:', config['feature'])
        print('filter_index', filter_index)
    filter_activation = activation[0][:, 1:-1, 1:-1, config['feature']]
    #plt.imshow(activation[:, 1:-1, 1:-1, filter_index][0])
    #print(activation[:, 1:-1, 1:-1, filter_index][0])
    return -tf.reduce_mean(filter_activation)


def define_generator(latent_dim = 100, regularization = None):
    if not regularization:
        regularization = 'NONE'
    input = keras.layers.Input(shape=[latent_dim])
    initial_shape = [1,1,1,128]
    
    final_shape = [10,8,8,3]
    x = keras.layers.Dense(initial_shape[0]*initial_shape[1]*initial_shape[2]*initial_shape[3],
                           )(input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Reshape(initial_shape)(x)
    #print(kernel)
    #8x8
    x = keras.layers.Conv3DTranspose(
        128, (2,2,2), strides=(2,2,2),padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #print(x.shape)
    #16*16
    x = keras.layers.Conv3DTranspose(
        128, (3,2,2), strides=(2,2,2),padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #print(x.shape)
    
    x = keras.layers.Conv3DTranspose(
        128, (2,2,2), strides=(2,2,2),padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #print(x.shape)
    #output layer
    x = keras.layers.Conv3DTranspose(
        3, (3,3,3), padding = 'same', activation = 'tanh')(x)
    #print(x.shape)
    x -= keras.backend.mean(x)
    x /= keras.backend.std(x) + 1e-5
    x *= 0.15

    # Center crop
    #img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    x += 0.5
    x = keras.backend.clip(x, 0, 1)
    model = keras.models.Model(inputs=[input],outputs=x, name = 'temo_spatial_generator')

    return model


def define_generator_spatial(latent_dim = 100, length = 10,regularization = None):
    if not regularization:
        regularization = 'NONE'
    input = keras.layers.Input(shape=[latent_dim])
    initial_shape = [1,1,128]
    
    final_shape = [1,8,8,3]
    x = keras.layers.Dense(initial_shape[0]*initial_shape[1]*initial_shape[2],
                           )(input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Reshape(initial_shape)(x)
    #print(x.shape)
    #8x8
    x = keras.layers.Conv2DTranspose(
        128, (2,2), strides=(2,2),padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #print(x.shape)
    #16*16
    x = keras.layers.Conv2DTranspose(
        128, (2,2), strides=(2,2), padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #print(x.shape)
    #32*32
    x = keras.layers.Conv2DTranspose(
    128, (2,2), strides=(2,2), padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #print(x.shape)
    #output layer
    x = keras.layers.Conv2DTranspose(
        3, (3,3), padding = 'same', activation = 'tanh')(x)
    #print(x.shape)
    x -= keras.backend.mean(x)
    x /= keras.backend.std(x) + 1e-5
    x *= 0.15

    x += 0.5
    x = keras.backend.clip(x, 0, 1)
    x = tf.expand_dims(x, axis = 1)
    old_x = x + 0
    for t in range(length-1):
        x = keras.layers.concatenate([x,old_x], axis = 1)
    #print(x.shape)
    model = keras.models.Model(inputs=[input],outputs=x, name = 'spatial_generator')
    


    return model

def define_generator_temporal(latent_dim = 100, regularization = None):
    if not regularization:
        regularization = 'NONE'
    input = keras.layers.Input(shape=[latent_dim])
    initial_shape = [1,128]
    
    final_shape = [10,8,8,3]
    x = keras.layers.Dense(initial_shape[0]*initial_shape[1],
                           )(input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Reshape(initial_shape)(x)
    #print(x.shape)
    #8x8
    x = keras.layers.Conv1DTranspose(
        128, (2), strides=(2),padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #x = keras.layers.Dropout(rate = 0.2)(x)
    #print(x.shape)
    #16*16
    x = keras.layers.Conv1DTranspose(
        128, (3), strides=(2), padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #x = keras.layers.Dropout(rate = 0.2)(x)
    #print(x.shape)
    #32*32
    x = keras.layers.Conv1DTranspose(
                            128, (2), strides=(2), padding = 'valid')(x)
    if regularization == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    if regularization[:7] == 'dropout':
        x = keras.layers.Dropout(float(regularization[-3:]))(x)
    #x = keras.layers.Dropout(rate = 0.2)(x)
    #print(x.shape)
    #output layer
    x = keras.layers.Conv1DTranspose(
        3, (3), padding = 'same', activation = 'tanh')(x)
    #print(x.shape)
    #x = keras.layers.LayerNormalization()(x)
    x -= keras.backend.mean(x)
    x /= keras.backend.std(x) + 1e-5
    x *= 0.15

    x += 0.5
    x = keras.backend.clip(x, 0, 1)
    x = tf.expand_dims(x, axis = 2)
    # x = tf.expand_dims(x, axis = 2)
    #print(x.shape)
    old_x = x + 0
    
    for w in range(8*8-1):
        x = keras.layers.concatenate([x,old_x], axis = 2)
    #print(x.shape)
    x = keras.layers.Reshape(final_shape)(x)
    # print(x.shape)
    model = keras.models.Model(inputs=[input],outputs=x, name = 'temporal_generator')


    return model
generator = define_generator()
optimizer = tf.keras.optimizers.Adam(lr=1e-2)
#%%

# def compute_loss(input_image, filter_index):
#     traject = np.array(create_trajectory((32,32),sample = 10, style = 'spiral'))#tf.random.uniform((10,2))
#     traject = np.expand_dims(traject, 0)
#     #input_image += tf.random.normal(input_image.shape, stddev = 0.005)
#     activation = feature_extractor((input_image,traject))
#     # We avoid border artifacts by only involving non-border pixels in the loss.
#     filter_activation = activation[0][:, 1:-1, 1:-1, filter_index]
#     #plt.imshow(activation[:, 1:-1, 1:-1, filter_index][0])
#     #print(activation[:, 1:-1, 1:-1, filter_index][0])
#     return -tf.reduce_mean(filter_activation)



def gradient_ascent_step(latent_starter, filter_index):
    with tf.GradientTape() as tape:
        img = generator(latent_starter)
        #tape.watch(generator)
        loss = compute_loss(filter_index,img)
 
    # Compute gradients.define_generator
    grads = tape.gradient(loss, generator.trainable_weights)
    #print(grads)
    # Normalize gradients.
    optimizer.apply_gradients(zip(grads, generator.trainable_weights))

    return loss, img


def visualize_filter(filter_index, use_img = False):
    # We run gradient ascent for 20 steps
    if config['test_mode']:
        iterations = 3
        
    else:
        iterations = config['gradient_ascent_iterations']
    loss_list = []
    latent_starter = tf.random.uniform([1,100])
    for iteration in range(iterations):
        #latent_starter = tf.random.uniform([1,100])
        loss, img = gradient_ascent_step(latent_starter,filter_index)
        loss_list.append(loss)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss_list, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    #img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # # Convert to RGB array
    # img *= 255
    # img = np.clip(img, 0, 255).astype("uint8")
    return img

#%%
# batch_size = 4 and 50 epochs takes 10min to train and outpits 1.72 stops training after 5 epochs. 
if config['testbad']:
    regularization_list = ['dropout0.2','dropout0.4','batchnorm', 'None']
    noise_list = [5e-2, 8e-2, 2e-1, 5e-1]
    img_list_test = []
    loss_list_test = []
    loss_data = pd.DataFrame(columns = ['epochs','loss','run_num','feature'])
    batch_size = 16
    dataframe = pd.DataFrame(columns = ['generator_type','regularization','noise','feature','epoch', 'loss'])
    num_examples = 5*32
    generators_list = [define_generator,define_generator_spatial, define_generator_temporal]
    if config['test_mode']:
        feature_num = 1
        epochs = 1
        repetitions = 2
    else:
        feature_num = 10
        epochs = 30
        repetitions = 10
    for generator_type in generators_list:
        print('############## ', generator_type, ' ####################')
        for i in range(feature_num):
            feature = [np.random.randint(0,64)]
            feature = np.array([num_examples*feature]).T
            config['feature'] = int(feature[0])
            print('#############    ','feature: ',feature[0],'    ##################')
            for regularization in regularization_list:
                for noise in noise_list:
                    print('############# ', regularization, ' ', noise, ' ###############')
                    config['noise'] = noise
                    
    
                    #feature = 42
    
                    #feature_list.append(feature)
                    #regular
                    temp_list = []
                    
                    for run in range(repetitions):
                        print(run)
                        generator = generator_type(latent_dim=100, regularization = regularization)
                        loss_func = compute_loss
                        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                        generator.compile(optimizer = optimizer,
                                          loss = loss_func,
                                          )
                       
                        latent_starter = list(np.random.uniform(size = [1,100]))
                        latent_starter = np.array(num_examples*latent_starter)
                        #print(time.strftime("%H:%M:%S"))
                        generator_history = generator.fit(x = latent_starter,
                                      y = feature,
                                      epochs = epochs,
                                      batch_size = batch_size,
                                      verbose = 0
                                      )
                        print('loss = :',generator_history.history['loss'][-1])
                        #print(time.strftime("%H:%M:%S"))
                        temp_list.append(-np.array(generator_history.history['loss']))
                        feature_name = epochs*[feature[0][0]]
                        dataframe = dataframe.append(pd.DataFrame({'generator_type': generator.name,
                                                                   'regularization':regularization,
                                                                   'noise':noise,'feature':feature_name,
                                                                   'epochs':np.arange(0,epochs),
                                                                   'loss':-np.array(generator_history.history['loss']),
                                                                   'final_loss' : generator_history.history['loss'][-1]}))
                    loss_list_test.append(temp_list)
    dataframe.to_pickle(dump_folder + '/testbad_results.pickle')
    dataframe.to_csv(dump_folder + '/testbad_results.csv')
    for generator_name in dataframe['generator_type'].unique():
        for feature in dataframe['feature'].unique():
            data = dataframe[dataframe['feature'] == feature]
            plt.figure()
            sns.boxplot(data = data, x = 'regularization', y = 'final_loss', hue = 'noise')
            plt.savefig(dump_folder + '/testbed_feature_{}_for_{}'.format(regularization[:3],generator_name))
    for generator_name in dataframe['generator_type'].unique():
        for regularization in dataframe['regularization'].unique():
            data = dataframe[dataframe['regularization'] == regularization]
            plt.figure()
            sns.boxplot(data = data, x = 'feature', y = 'final_loss', hue = 'noise')
            plt.savefig(dump_folder + '/testbed_{}_for_{}'.format(regularization[:3],generator_name))
    #     plt.figure()
    #     for l in temp_list:
    #         plt.plot(l)
    #     feature_name = feature
    # #         img_list_test.append(generator.predict(latent_starter[:2])[0])
    #         # generator = define_generator_temporal(latent_dim=100)
    #         # optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    #         # loss, img = visualize_filter(feature[0][0])
    #         # 
    #         # #img_list.append(img)
            
    #         # print(loss[-1])
    #         # print(time.strftime("%H:%M:%S"))
    #         # temp_dataframe = pd.DataFrame({'epochs':np.arange(0,500), 
    #         #                                 'loss' : list(-np.array(loss)),
    #         #                                 'norm_loss' : list((-np.array(loss)  - np.mean(-np.array(loss)))/np.std(-np.array(loss))),
    #         #                                 'run_num': np.array([run] * 500 ),
    #         #                                 'feature':feature})
    #         # loss_data = loss_data.append(temp_dataframe)
    #         # fig, ax = plt.subplots(3,3)
    #         # fig.suptitle(feature)
    #         # indx = 0
    # #     # for l in range(3):
    # #     #     for k in range(3):
    # #     #         ax[l,k].imshow(img[indx,:,:,:])
    # #     #         indx+=1
    # #     #         ax[l,k].title.set_text(indx)
    # pickle.dump(loss_list_test, open('loss_list_test', 'wb'))
    # # # plt.figure()
    # # for loss_i in loss_list_test:
    # #     plt.plot(loss_i)
    # # plt.title(feature)
    #%%
    # plt.figure()
    # sns.lineplot(data = dataframe, x = 'epochs', y = 'loss')
    # plt.title('Spacio-Temporal Activation Maximization Values During Training')
    # plt.xlabel('Epoch')
    # plt.ylabel('arb. unit')
    # #sns.lineplot(data = silance_slice, x = 'num_silanced', y = 'delta_accur', hue = 'type', marker = 'o')
    # plt.ylim(-1,1)
else:
#%%
    if config['find_AM']:
        # The dimensions of our input image
        img_width = 8
        img_height = 8
        loss_list = []
        loss_list_spatial = []
        loss_list_temporal = []
        shuffle = []
        spatial_shuffle = []
        temporal_shuffle = []
        img_list = {}
        img_list_spatial = {}
        feature_list = []
        img_list_temporal = {}
        latent_dim = 100
        st_loss_vals = pd.DataFrame(columns = ['feature','mean','std','CV','last_loss'])
        s_loss_vals = pd.DataFrame()
        t_loss_vals = pd.DataFrame()
        num_features = 64
        num_examples = 5*32
        batch_size = 16
        for i in range(num_features):
            print('#############    ',i,'    ##################')
            feature = [i]
            feature = np.array([num_examples*feature]).T
            feature_list.append(feature)
            config['feature'] = int(feature[0])
            #regular
            feature_losses = []
            feature_images = []
            image_count = 0 #limit saving images to 3 to decrease memory usage
            for run in range(config['num_of_runs']):
                generator = define_generator(latent_dim=100, regularization = config['regularization'])
                loss_func = compute_loss
                optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                generator.compile(optimizer = optimizer,
                                  loss = loss_func,
                                  )
               
                latent_starter = list(np.random.uniform(size = [1,100]))
                latent_starter = np.array(num_examples*latent_starter)
                if config['loud']:
                    print(time.strftime("%H:%M:%S"))
                generator_history = generator.fit(x = latent_starter,
                              y = feature,
                              epochs = config['gradient_ascent_iterations'],
                              batch_size = batch_size,
                              verbose = 0
                              )
                if config['loud']:
                    print(time.strftime("%H:%M:%S"))
        
                if image_count < 3:
                    feature_images.append(generator(latent_starter[:2])[0].numpy())
                    image_count += 1
                #img_list.append(img)
                feature_losses.append(-np.array(generator_history.history['loss']))
            img_list[feature[0][0]] = feature_images
            loss_list.append(feature_losses)
            last_loss = np.array(feature_losses)[:,-1]
            temp_st_dict = {'feature':i,
                            'mean': np.mean(last_loss),
                            'std' : np.std(last_loss),
                            'CV'  : np.std(last_loss)/np.mean(last_loss),
                            'last_loss': last_loss}
            print('spatio-remporal: ',temp_st_dict)
            st_loss_vals = st_loss_vals.append(temp_st_dict,ignore_index=True)
            ##########################################################################
            #spatial
            feature_losses = []
            feature_images = []
            image_count = 0
            for run in range(config['num_of_runs']):
                generator = define_generator_spatial(latent_dim=100, regularization = config['regularization'])
                loss_func = compute_loss
                optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                generator.compile(optimizer = optimizer,
                                  loss = loss_func,
                                  )
               
                latent_starter = list(np.random.uniform(size = [1,100]))
                latent_starter = np.array(num_examples*latent_starter)
                if config['loud']:
                    print(time.strftime("%H:%M:%S"))
                generator_history = generator.fit(x = latent_starter,
                              y = feature,
                              epochs = config['gradient_ascent_iterations'],
                              batch_size = batch_size,
                              verbose = 0
                              )
                if config['loud']:
                    print(time.strftime("%H:%M:%S"))
        
                if image_count < 3:
                    feature_images.append(generator(latent_starter[:2])[0].numpy())
                    image_count += 1
                #img_list.append(img)
                feature_losses.append(-np.array(generator_history.history['loss']))
            img_list_spatial[feature[0][0]] = feature_images
            loss_list_spatial.append(feature_losses)
            last_loss = np.array(feature_losses)[:,-1]
            temp_s_dict = {'feature':i,
                            'mean': np.mean(last_loss),
                            'std' : np.std(last_loss),
                            'CV'  : np.std(last_loss)/np.mean(last_loss),
                            'last_loss': last_loss}
            print('spatial: ',temp_s_dict)
            s_loss_vals = s_loss_vals.append(temp_s_dict,ignore_index=True)
            ##########################################################################
            #temporal
            feature_losses = []
            feature_images = []
            image_count = 0
            for run in range(config['num_of_runs']):
                generator = define_generator_temporal(latent_dim=100, regularization = config['regularization'])
                loss_func = compute_loss
                optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                generator.compile(optimizer = optimizer,
                                  loss = loss_func,
                                  )
               
                latent_starter = list(np.random.uniform(size = [1,100]))
                latent_starter = np.array(num_examples*latent_starter)
                if config['loud']:
                    print(time.strftime("%H:%M:%S"))
                generator_history = generator.fit(x = latent_starter,
                              y = feature,
                              epochs = config['gradient_ascent_iterations'],
                              batch_size = batch_size,
                              verbose = 0
                              )
                if config['loud']:
                    print(time.strftime("%H:%M:%S"))
                if image_count < 3:
                    feature_images.append(generator(latent_starter[:2])[0].numpy())
                    image_count += 1
                #img_list.append(img)
                feature_losses.append(-np.array(generator_history.history['loss']))
            img_list_temporal[feature[0][0]] = feature_images
            loss_list_temporal.append(feature_losses)
            last_loss = np.array(feature_losses)[:,-1]
            temp_t_dict = {'feature':i,
                            'mean': np.mean(last_loss),
                            'std' : np.std(last_loss),
                            'CV'  : np.std(last_loss)/np.mean(last_loss),
                            'last_loss': last_loss}
            print('temporal: ',temp_t_dict)
            t_loss_vals = t_loss_vals.append(temp_t_dict,ignore_index=True)
            
        pickle.dump(img_list,open(dump_folder + '/img_st_{}'.format(run_name),'wb'))
        pickle.dump(img_list_spatial,open(dump_folder + '/img_s_{}'.format(run_name),'wb'))
        pickle.dump(img_list_temporal,open(dump_folder + '/img_t_{}'.format(run_name),'wb'))
        
        loss_t = np.array(loss_list_temporal)
        loss_s = np.array(loss_list_spatial)
        loss_st = np.array(loss_list)
        
        pickle.dump(loss_st,open(dump_folder + '/loss_st_{}'.format(run_name),'wb'))
        pickle.dump(loss_s,open(dump_folder + '/loss_s_{}'.format(run_name),'wb'))
        pickle.dump(loss_t,open(dump_folder + '/loss_t_{}'.format(run_name),'wb'))
        
        st_loss_vals = st_loss_vals.sort_values(['feature'])
        s_loss_vals = s_loss_vals.sort_values(['feature'])
        t_loss_vals = t_loss_vals.sort_values(['feature'])
        st_loss_vals.to_pickle(dump_folder + '/loss_st_dataframe')
        s_loss_vals.to_pickle(dump_folder + '/loss_s_dataframe')
        t_loss_vals.to_pickle(dump_folder + '/loss_t_dataframe')
        # plt.figure()
        # plt.plot(loss_t,loss_s,'o')
        # plt.xlabel('Purely Temporal Activation')
        # plt.ylabel('Purly Spatial Activation')
        
        
        plt.figure()
        scatter_data = pd.DataFrame({'st':st_loss_vals['mean'], 's':s_loss_vals['mean'],'t':t_loss_vals['mean']})
        sns.scatterplot(data=scatter_data, x="t", y="s", hue="st", size="st",sizes=(40, 140))
        plt.legend(bbox_to_anchor=(-0.15, 1), loc=0, borderaxespad=0.,title = 'Spatio-temporal\n Activation',fontsize = 10, title_fontsize=12)
        plt.xlabel('Purely Temporal Activation', fontsize = 16)
        plt.ylabel('Purly Spatial Activation', fontsize = 16)
        plt.savefig(dump_folder + '/t_vs_s_scatter_0.png')
        #plt.xticks([0,4,8],fontsize = 14)
        #plt.yticks([1,3,6], fontsize = 14)
        plt.figure()
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=2)
        sns.scatterplot(data=scatter_data, x="t", y="s", hue="st", size="st",sizes=(40, 140))
        plt.legend(bbox_to_anchor=(1.02, 1), loc=0, borderaxespad=0.,fontsize = 14)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.savefig(dump_folder + '/t_vs_s_scatter.png')
    
    else:
        t_loss_vals = pd.read_pickle(config['AM_list_path'] + '/loss_t_dataframe')
        s_loss_vals = pd.read_pickle(config['AM_list_path'] + '/loss_s_dataframe')
        st_loss_vals = pd.read_pickle(config['AM_list_path'] + '/loss_st_dataframe')
        st_loss_vals = st_loss_vals.sort_values(['feature'])
        s_loss_vals = s_loss_vals.sort_values(['feature'])
        t_loss_vals = t_loss_vals.sort_values(['feature'])
        scatter_data = pd.DataFrame({'st':st_loss_vals['mean'], 's':s_loss_vals['mean'],'t':t_loss_vals['mean']})
    #%%
    if config['feature_importance']:
        ################################################################################
        ##################### Evaluate the impact of silancing the #####################
        #####################    t, s and ts sensative neurons     #####################
        ################################################################################
        model_copy = keras.models.clone_model(student)
        model_copy.set_weights(student.get_weights()) 
        opt=tf.keras.optimizers.Adam(lr=1e-3)
        feature_importance_metrix = np.zeros([11, 7, 11])
        feature_percentage_metrix = np.zeros([11, 7, 10])
        
        def silence_feature(model_copy, orig_model, index, layer_name = config['student_AM_layer']):
            weights = np.array(orig_model.get_layer(layer_name).weights[1])
            if config['feature_silencing_tech'] == 'zero':
                weights[:,:,index, :] = 0
            elif config['feature_silencing_tech'] == 'shuffle':
                weights[:,:,index, :] = np.random.permutation(weights[:,:,index,:])
            elif config['feature_silencing_tech'] == 'rendom':
                weights[:,:,index, :] = np.random.normal(loc = np.mean(weights[:,:,index, :]),scale = np.std(weights[:,:,index, :] ), size = weights[:,:,index,:].shape)
            new_weights = [np.array(orig_model.get_layer(layer_name).weights[0]),weights, np.array(orig_model.get_layer(layer_name).weights[2])]
            model_copy.get_layer(layer_name).set_weights(new_weights)
            return model_copy
        
        #################### Go feature by feature from the strongest ##################
        #################### to the weakest and record it's effect on ################## 
        ####################               accuracy                   ##################
        
        
        ####################         Call Dataset Generator           ##################
        # load dataset
        (trainX, trainY), (testX, testY) = cifar10.load_data()
        def prep_pixels(train, test,resnet_mode=False):
            # convert from integers to floats
            if resnet_mode:
                train_norm = cifar10_resnet50.preprocess_image_input(train)
                test_norm = cifar10_resnet50.preprocess_image_input(test)
                print('preprocessing in resnet mode')
            else:
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
        trainX, testX = prep_pixels(trainX, testX, resnet_mode=parameters['resnet_mode'])
        
        def args_to_dict(**kwargs):
            return kwargs
        position_dim = (int(parameters['max_length']),int(parameters['res']),int(parameters['res']),2) if  parameters['broadcast']==1 else (int(parameters['max_length']),2)
        movie_dim = (int(parameters['max_length']), int(parameters['res']), int(parameters['res']), 3)
            
        BATCH_SIZE=32
        ctrl_mode = parameters['student_version'] > 100
        position_dim = (parameters['max_length'],parameters['res'],parameters['res'],2) if  parameters['broadcast']==1 else (parameters['n_samples'],2)
        movie_dim = (parameters['max_length'], parameters['res'], parameters['res'], 3)  if parameters['student_version'] < 100 else (parameters['res'], parameters['res'], 3)
        def args_to_dict(**kwargs):
            return kwargs
        generator_params = args_to_dict(    one_random_sample=ctrl_mode,
                                            batch_size=BATCH_SIZE, movie_dim=movie_dim, position_dim=position_dim, n_classes=None, shuffle=False,
                                            prep_data_per_batch=True,one_hot_labels=False,
                                            res = parameters['res'],
                                            n_samples = parameters['n_samples'],
                                            mixed_state = True,
                                            n_trajectories = parameters['trajectories_num'],
                                            trajectory_list = 0,
                                            broadcast=parameters['broadcast'],
                                            style = parameters['style'],
                                            max_length=parameters['max_length'],
                                            noise = parameters['noise'],
                                            time_sec=parameters['time_sec'], 
                                            traj_out_scale=parameters['traj_out_scale'],  
                                            snellen=parameters['snellen'],
                                            vm_kappa=parameters['vm_kappa'],
                                            random_n_samples = parameters['random_n_samples'],
                                            syclopic_norm=parameters['syclopic_norm'],
                                            shuffle_traj=parameters['shuffle_traj']
        )

        
        #generator_params = {'one_random_sample': False, 'batch_size': 32, 'movie_dim': (10, 8, 8, 3), 'position_dim': (10, 2), 'n_classes': None, 'shuffle': False, 'prep_data_per_batch': True, 'one_hot_labels': False, 'res': 8, 'n_samples': 10, 'mixed_state': True, 'n_trajectories': -1, 'trajectory_list': 0, 'broadcast': 0, 'style': 'spiral_2dir2', 'max_length': 10, 'noise': 0.5, 'time_sec': 0.3, 'traj_out_scale': 4.0, 'snellen': True, 'vm_kappa': 0.0, 'random_n_samples': 0, 'syclopic_norm': 1.0, 'shuffle_traj': False}
        print(generator_params)
        print('preparing generators')
        
        
        
        if config['test_mode']:
            val_generator_classifier = Syclopic_dataset_generator(
                                    trainX[-100:].repeat(parameters['val_set_mult'],axis=0), 
                                    trainY[-100:].repeat(parameters['val_set_mult'],axis=0), 
                                    validation_mode=True,
                                    **generator_params)
            train_generator_classifier = Syclopic_dataset_generator(
                                    trainX[:500], 
                                    trainY[:500], 
                                    **generator_params)
        else:
            train_generator_classifier = Syclopic_dataset_generator(
                                    trainX[:-5000], 
                                    trainY[:-5000], 
                                    **generator_params)
            val_generator_classifier = Syclopic_dataset_generator(testX, testY,validation_mode=True, **generator_params)
        # train_generator_features = Syclopic_dataset_generator(trainX[:-5000], None, teacher=fe_model, **generator_params)
        # val_generator_features = Syclopic_dataset_generator(trainX[-5000:].repeat(parameters['val_set_mult'],axis=0), None, teacher=fe_model, validation_mode=True, **generator_params)
        
        #%%
        
        #print('The Decoder Baseline Accuracy = ', parameters['decoder_max_test_error'])
        #########################       Start Evaluation      ################################
        ################################################################################
        ######################### Combine Student and Decoder ##########################
        ################################################################################
        if config['loading_style'] == 'direct':
            print('Evaluating network with current data (chacking network was loaded correctly) - direct loading')
            full_accur = student_and_decoder.evaluate(val_generator_classifier, verbose = 2)[1]
        model_copy.trainable = False
        # config = student.get_config() # Returns pretty much every information about your model
        input0 = keras.layers.Input(shape=movie_dim)
        input1 = keras.layers.Input(shape=position_dim)
        x = model_copy((input0,input1))
        x = decoder(x)
        student_and_decoder = keras.models.Model(inputs=[input0,input1], outputs=x, name='DRC')
        student_and_decoder.compile(
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"],
            )
        #%%
        ####################### Evaluate and retrain decoder ###########################
        print('Evaluating network with current data (chacking network was loaded correctly)')
        full_accur = student_and_decoder.evaluate(val_generator_classifier, verbose = 2)[1]
        
        TESTMODE = config['test_mode']
        if config['train_decoder']:
            ################### Evaluate with Student Features #########################
            lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            early_stopper = keras.callbacks.EarlyStopping(
                monitor='val_sparse_categorical_accuracy', min_delta=1e-4, patience=10, verbose=0,
                mode='auto', baseline=None, restore_best_weights=True
            )
            
            parameters['pre_training_decoder_accur'] = full_accur
            ################## Re-train the half_net with the ##########################
            ##################   student training features    ##########################        
            print('\nTraining the decoder')
            decoder_history = student_and_decoder.fit(train_generator_classifier,
                                   epochs = int(parameters['decoder_epochs']) if not TESTMODE else 1,
                                   validation_data = val_generator_classifier,
                                   verbose = 2,
                                   callbacks=[lr_reducer,early_stopper],
                        workers=8, use_multiprocessing=True)
        #%%
        ######################         get list of           ########################### 
        ######################     strongest to weakest      ###########################
        ######################         activations           ###########################
        val_to_kick = np.array(scatter_data['t']).argsort()[::-1][0]
        s_to_w_t = np.array(scatter_data['t']).argsort()[::-1][:-1]
        s_to_w_s = np.array(scatter_data['s']).argsort()[::-1][:-1]
        s_to_w_st = np.array(scatter_data['st']).argsort()[::-1][:-1]
        print('Top Spatio-Temporal features - ', s_to_w_st[:10])
        print('Top Spatial features - ', s_to_w_s[:10])
        print('Top Temporal features - ', s_to_w_t[:10])
        #%%
        if config['one_by_one_slice']:
            ######################  Evaluate change in accuracy  ###########################  
            accur_after_zero = []
            for i in range(64):
                print(i)
                model_copy = silence_feature( model_copy = model_copy,
                                                 orig_model = student, 
                                                 index = i)
                ############################################################################
                ############################# Combine Student and Decoder ##################
                ############################################################################
                model_copy.trainable = False
                # config = student.get_config() # Returns pretty much every information about your model
                input0 = keras.layers.Input(shape=movie_dim)
                input1 = keras.layers.Input(shape=position_dim)
                x = model_copy((input0,input1))
                x = decoder(x)
                student_and_decoder = keras.models.Model(inputs=[input0,input1], outputs=x, name='DRC')
                
                
                student_and_decoder.compile(
                        loss="sparse_categorical_crossentropy",
                        metrics=["sparse_categorical_accuracy"],
                    )
                accur = student_and_decoder.evaluate(val_generator_classifier, verbose = 2)[1]
                accur_after_zero.append(accur)
            #%%
            #Make a dataframe holding the accuracoes by order of st, s, t 
            after_zero_dataframe = pd.DataFrame(columns = ['st','s','t'])
            for i in range(0):
                new_row = {'st':accur_after_zero[s_to_w_st[i]],
                           's':accur_after_zero[s_to_w_s[i]],
                           't':accur_after_zero[s_to_w_t[i]]}
                after_zero_dataframe = after_zero_dataframe.append(new_row, ignore_index=True)
            after_zero_dataframe.to_pickle(dump_folder + '/after_zero_dataframe_{}'.format(run_name))
        #%%
        ########################    Get the top AM with        #########################
        ########################    with limits on the         #########################
        ########################    other AM                   #########################
        plt.figure()
        plt.scatter(t_loss_vals['mean'], s_loss_vals['mean'])
        # Get strength lists
        all_position_data = pd.DataFrame()
        all_position_data = pd.DataFrame({'st':s_to_w_st,'s':s_to_w_s,'t':s_to_w_t})
        top_t_vals = []
        top_s_vals = []
        top_t_s_vals = []
        top_s_t_vals = []
        top_t_features = []
        top_s_features = []
        s_full = False
        t_full = False
        for feature in all_position_data['t']: 
            if not t_full:
                print(feature)
                if s_loss_vals.iloc[feature]['mean'] < config['s_lim']:
                    top_t_vals.append(t_loss_vals.iloc[feature]['mean'])
                    top_t_s_vals.append(s_loss_vals.iloc[feature]['mean'])
                    top_t_features.append(feature)
                if len(top_t_vals) == 15:
                    t_full = True
        for feature in all_position_data['s']: 
            if not s_full:
                if t_loss_vals.iloc[feature]['mean'] < config['t_lim']:
                    top_s_vals.append(s_loss_vals.iloc[feature]['mean'])
                    top_s_t_vals.append(t_loss_vals.iloc[feature]['mean'])
                    top_s_features.append(feature)
                if len(top_s_vals) == 15:
                    s_full = True
        plt.scatter(top_t_vals[:10],top_t_s_vals[:10])
        plt.scatter(top_s_t_vals[:10],top_s_vals[:10])
        #plt.xlim(-0.05,0.8)
        #plt.ylim(-0.05,0.4)
        plt.savefig(dump_folder + '/scatter_of silanced_features.png')
        print('top Temporal features with SPATIAL AM lower then {} are {}'.format(config['s_lim'], top_t_features))
        print('top Spatial features with TEMPORAL AM lower then {} are {}'.format(config['t_lim'], top_s_features))
        #%%
        ########################     Evaluating 1 to 10        ######################### 
        ########################     strongest features        #########################
        ########################        of t, s & st           #########################
        all_ratings = [s_to_w_st[config['outlier']:], top_s_features, top_t_features, np.random.randint]
        all_names = ['tempo_spatial','spatial','temporal', 'both'] + config['num_random_evaluations']*['random']
        silance_slice = pd.DataFrame(columns = ['num_silanced','type','accur','delta_accur'])
        #Going one by one - st, s, t - and silancing the 5 - 10 - 20 most activated features.
        full_accur = student_and_decoder.evaluate(val_generator_classifier, verbose = 2)[1]
        
        for top in [2,4,6,8,10]:#,13,15,17,20]:
            temp_row = []
            random_cach = pd.DataFrame(columns = ['num_silanced','type','accur','delta_accur'])
            print('####################### ', top, ' ################################')
            for indx, name in enumerate(all_names):
                print('--------- ',all_names[indx],' ----------')
                if name == 'random':
                    choose_from = np.arange(0,64)
                    choose_from = np.delete(choose_from, val_to_kick)
                    mode = np.random.choice(choose_from,size = top, replace = False)
                        
                elif name == 'both':
                    mode1 = top_s_features
                    mode2 = top_t_features
                    mode = mode1[:top//2] + mode2[:top//2]
                else:
                    mode = all_ratings[indx]
                print(name,mode)
                model_copy = keras.models.clone_model(student)
                model_copy.set_weights(student.get_weights()) 
                weights = np.array(model_copy.get_layer(config['student_AM_layer']).weights[1])
                mean_weights_val = []
                for k in range(top):
                    mean_weights_val.append(weights[:,:,mode[k],:].mean())
                print('mean weights of the silanced features - before ',mean_weights_val)
                for k in range(top):
                    print('silencing feature {}'.format(mode[k]))
                    model_copy = silence_feature( model_copy = model_copy,
                                             orig_model = model_copy, 
                                         index = mode[k])
                weights = np.array(model_copy.get_layer(config['student_AM_layer']).weights[1])
                mean_weights_val = []
                for k in range(top):
                    mean_weights_val.append(weights[:,:,mode[k],:].mean())
                print('mean weights of the silanced features - after ',mean_weights_val)
                ############################################################################
                ############################# Combine Student and Decoder ##################
                ############################################################################
                model_copy.trainable = False
                # config = student.get_config() # Returns pretty much every information about your model
                input0 = keras.layers.Input(shape=movie_dim)
                input1 = keras.layers.Input(shape=position_dim)
                x = model_copy((input0,input1))
                x = decoder(x)
                student_and_decoder = keras.models.Model(inputs=[input0,input1], outputs=x, name='DRC')
                
                
                student_and_decoder.compile(
                        loss="sparse_categorical_crossentropy",
                        metrics=["sparse_categorical_accuracy"],
                    )
                
                accur = student_and_decoder.evaluate(val_generator_classifier, verbose = 2)[1]
                temp_row = {'num_silanced':top, 'type':all_names[indx], 'accur':accur, 'delta_accur':full_accur - accur}
                if name == 'random':
                    random_cach = random_cach.append(temp_row, ignore_index = True)
                    print('When silancing the top {} in {} we get a {} decrease in accuracy'.format(top, all_names[indx], full_accur - accur))
                else:
                    print('When silancing the top {} in {} we get a {} decrease in accuracy'.format(top, all_names[indx], full_accur - accur))
                    silance_slice = silance_slice.append(temp_row, ignore_index = True)
            print('When silancing the top {} in {} we get an avarage of {} decrease in accuracy'.format(top, all_names[indx], random_cach['accur'].mean() - accur))
            temp_row = {'num_silanced':top, 'type':name, 'accur':random_cach['accur'].mean(), 'delta_accur':full_accur - random_cach['accur'].mean()}  
            silance_slice = silance_slice.append(temp_row, ignore_index = True)
        silance_slice.to_pickle(dump_folder + '/silance_slice_{}'.format(run_name))
        plt.figure()
        sns.barplot(data=silance_slice, x='num_silanced', y="delta_accur", hue="type")
        plt.savefig(dump_folder + '/accuracy_drop_by_features.png')
        #%%
        plt.figure()
        sns.lineplot(data = silance_slice, x = 'num_silanced', y = 'delta_accur', hue = 'type', marker = 'o')
        plt.savefig(dump_folder + '/accuracy_drop_lineplot.png')