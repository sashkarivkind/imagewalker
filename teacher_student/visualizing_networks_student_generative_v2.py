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
print(os.getcwd() + '/')

parser = argparse.ArgumentParser()

#general parameters
parser.add_argument('--path', default=os.getcwd()+'/', help = 'the path from where to take the student and save the data')
parser.add_argument('--run_name', default = 'noname_j96976_t1636556927')
parser.add_argument('--test_mode', default=0, type = int, help = 'if True will run over 10 features to test')
parser.add_argument('--train_decoder', default=1, type = int, help = 'RETRAIN DECODER - to turn off if we fix the saving issue')
parser.add_argument('--num_of_runs', default = 5, type = int, help = 'How many times to run the generator over a single feature to get distrebution of activation values')
parser.add_argument('--feature_silencing_tech', default = 'random', help = 'What techniuqe to use in order to silance out a feature - options: zero, shuffle, random')
config = parser.parse_args()
config = vars(config)
print('config  ',config)
#%%
############################### Get Trained Student ##########################3
run_name = config['run_name']
num_samples = 0
dump_folder = config['path'] + run_name + '_{}'.format(time.strftime("%m-%d-%H_%M_%S"))
config['dump_folder'] = dump_folder
os.mkdir(dump_folder)
student = None
student, parameters, decoder = load_student(path = config['path'], 
                                           run_name = run_name, 
                                           num_samples = num_samples,
                                           student=student)
config['parameters'] = parameters
run_name = run_name + '_{}'.format(time.strftime("%m_%d_%H_%M"))

#%%

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
layer = student.get_layer(name='convLSTM30')
feature_extractor = keras.Model(inputs=student.inputs, outputs=layer.output)

#%%
def define_generator(latent_dim = 100, res_net = True):
    input = keras.layers.Input(shape=[latent_dim])
    initial_shape = [1,1,1,128]
    
    if res_net:
        kernel = 1
    else:
        kernel = 2
    final_shape = [10,8,8,3]
    x = keras.layers.Dense(initial_shape[0]*initial_shape[1]*initial_shape[2]*initial_shape[3],
                           )(input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Reshape(initial_shape)(x)
    #print(kernel)
    #8x8
    x = keras.layers.Conv3DTranspose(
        128, (2,2,2), strides=(2,2,2),padding = 'valid')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    #print(x.shape)
    #16*16
    x = keras.layers.Conv3DTranspose(
        128, (3,2,2), strides=(2,2,2),padding = 'valid')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    #print(x.shape)
    
    x = keras.layers.Conv3DTranspose(
        128, (2,2,2), strides=(2,2,2),padding = 'valid')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
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
    model = keras.models.Model(inputs=[input],outputs=x, name = 'student_3')


    return model


def define_generator_spatial(latent_dim = 100, length = 10):
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
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    #print(x.shape)
    #16*16
    x = keras.layers.Conv2DTranspose(
        128, (2,2), strides=(2,2), padding = 'valid')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    #print(x.shape)
    #32*32
    x = keras.layers.Conv2DTranspose(
    128, (2,2), strides=(2,2), padding = 'valid')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
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
    model = keras.models.Model(inputs=[input],outputs=x, name = 'student_3')


    return model

def define_generator_temporal(latent_dim = 100):
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
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    #x = keras.layers.Dropout(rate = 0.2)(x)
    #print(x.shape)
    #16*16
    x = keras.layers.Conv1DTranspose(
        128, (3), strides=(2), padding = 'valid')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    #x = keras.layers.Dropout(rate = 0.2)(x)
    #print(x.shape)
    #32*32
    x = keras.layers.Conv1DTranspose(
    128, (2), strides=(2), padding = 'valid')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
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
    model = keras.models.Model(inputs=[input],outputs=x, name = 'student_3')


    return model
generator = define_generator()
optimizer = tf.keras.optimizers.Adam(lr=1e-2)
#%%
def compute_loss(input_image, filter_index):
    traject = np.array(create_trajectory((32,32),sample = 10, style = 'spiral'))#tf.random.uniform((10,2))
    traject = np.expand_dims(traject, 0)
    input_image += tf.random.normal(input_image.shape, stddev = 0.005)
    activation = feature_extractor((input_image,traject))
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[0][:, 1:-1, 1:-1, filter_index]
    #plt.imshow(activation[:, 1:-1, 1:-1, filter_index][0])
    #print(activation[:, 1:-1, 1:-1, filter_index][0])
    return -tf.reduce_mean(filter_activation)



def gradient_ascent_step(latent_starter, filter_index):
    with tf.GradientTape() as tape:
        img = generator(latent_starter)
        #tape.watch(generator)
        loss = compute_loss(img, filter_index)
 
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
        iterations = 500
    loss_list = []
    latent_starter = tf.random.uniform([1,100])
    for iteration in range(iterations):
        #latent_starter = tf.random.uniform([1,100])
        loss, img = gradient_ascent_step(latent_starter, filter_index)
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
# img_list_test = []
# loss_list_test = []
# loss_data = pd.DataFrame(columns = ['epochs','loss','run_num','feature'])
# for i in range(3):
#     feature = np.random.randint(0,64)
#     #feature = 42
#     print('#############    ','feature: ',feature,'    ##################')
    
#     #feature_list.append(feature)
#     #regular
#     for run in range(10):
#         print(run)
#         generator = define_generator(latent_dim=100)
#         optimizer = tf.keras.optimizers.Adam(lr=1e-3)
#         loss, img = visualize_filter(feature)
#         img_list_test.append(img)
#         #img_list.append(img)
#         loss_list_test.append(-np.array(loss))
#         temp_dataframe = pd.DataFrame({'epochs':np.arange(0,500), 
#                                         'loss' : list(-np.array(loss)),
#                                         'norm_loss' : list((-np.array(loss)  - np.mean(-np.array(loss)))/np.std(-np.array(loss))),
#                                         'run_num': np.array([run] * 500 ),
#                                         'feature':feature})
#         loss_data = loss_data.append(temp_dataframe)
#         # fig, ax = plt.subplots(3,3)
#         # fig.suptitle(feature)
#         # indx = 0
#     # for l in range(3):
#     #     for k in range(3):
#     #         ax[l,k].imshow(img[indx,:,:,:])
#     #         indx+=1
#     #         ax[l,k].title.set_text(indx)

# # plt.figure()
# # for loss_i in loss_list_test:
# #     plt.plot(loss_i)
# # plt.title(feature)
# plt.figure()
# sns.lineplot(data = loss_data, x = 'epochs', y = 'loss', hue='feature')
# plt.title('Spacio-Temporal Activation Maximization Values During Training')
# plt.xlabel('Epoch')
# plt.ylabel('arb. unit')
# #sns.lineplot(data = silance_slice, x = 'num_silanced', y = 'delta_accur', hue = 'type', marker = 'o')
# plt.ylim(-1,1)
#%%

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
for i in range(num_features):
    print('#############    ',i,'    ##################')
    feature = i
    feature_list.append(feature)
    #regular
    feature_losses = []
    feature_images = []
    image_count = 0 #limit saving images to 3 to decrease memory usage
    for run in range(config['num_of_runs']):
        generator = define_generator(latent_dim=latent_dim)
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        loss, img = visualize_filter(feature)
        if image_count < 3:
            feature_images.append(img)
            image_count += 1
        #img_list.append(img)
        feature_losses.append(-np.array(loss))
    img_list[feature] = feature_images
    loss_list.append(feature_losses)
    last_loss = np.array(feature_losses)[:,-1]
    temp_st_dict = {'feature':i,
                    'mean': np.mean(last_loss),
                    'std' : np.std(last_loss),
                    'CV'  : np.std(last_loss)/np.mean(last_loss),
                    'last_loss': last_loss}
    print(temp_st_dict)
    st_loss_vals = st_loss_vals.append(temp_st_dict,ignore_index=True)
    ##########################################################################
    #spatial
    feature_losses = []
    feature_images = []
    image_count = 0
    for run in range(config['num_of_runs']):
        generator = define_generator_spatial(latent_dim=latent_dim)
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        loss, img = visualize_filter(feature)
        if image_count < 3:
            feature_images.append(img)
            image_count += 1
        feature_losses.append(-np.array(loss))
    img_list_spatial[feature] = feature_images
    loss_list_spatial.append(feature_losses)
    last_loss = np.array(feature_losses)[:,-1]
    temp_s_dict = {'feature':i,
                    'mean': np.mean(last_loss),
                    'std' : np.std(last_loss),
                    'CV'  : np.std(last_loss)/np.mean(last_loss),
                    'last_loss': last_loss}
    print(temp_s_dict)
    s_loss_vals = s_loss_vals.append(temp_s_dict,ignore_index=True)
    ##########################################################################
    #temporal
    feature_losses = []
    feature_images = []
    image_count = 0
    for run in range(config['num_of_runs']):
        generator = define_generator_temporal(latent_dim=latent_dim)
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        loss, img = visualize_filter(feature)
        if image_count < 3:
            feature_images.append(img)
            image_count += 1
        feature_losses.append(-np.array(loss))
    img_list_temporal[feature] = feature_images
    loss_list_temporal.append(feature_losses)
    last_loss = np.array(feature_losses)[:,-1]
    temp_t_dict = {'feature':i,
                    'mean': np.mean(last_loss),
                    'std' : np.std(last_loss),
                    'CV'  : np.std(last_loss)/np.mean(last_loss),
                    'last_loss': last_loss}
    print(temp_t_dict)
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
st_loss_vals.to_pickle(dump_folder + '/loss_st_dataframe', protocol=3)
s_loss_vals.to_pickle(dump_folder + '/loss_s_dataframe', protocol=3)
t_loss_vals.to_pickle(dump_folder + '/loss_t_dataframe', protocol=3)
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


#%%
#load old loss
# loss_t = pickle.load(open('loss_t_noname_j956187_t1633205328_0', 'rb'))
# loss_s = pickle.load(open('loss_s_noname_j956187_t1633205328_0', 'rb'))
# loss_st = pickle.load(open('loss_st_noname_j956187_t1633205328_0', 'rb'))
#%%
################################################################################
##################### Evaluate the impact of silancing the #####################
#####################    t, s and ts sensative neurons     #####################
################################################################################

model_copy = keras.models.clone_model(student)
model_copy.set_weights(student.get_weights()) 
opt=tf.keras.optimizers.Adam(lr=1e-3)
feature_importance_metrix = np.zeros([11, 7, 11])
feature_percentage_metrix = np.zeros([11, 7, 10])

def silence_feature(model_copy, orig_model, index, layer_name = 'convLSTM30'):
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
        train_norm = train_norm / 1.0
        test_norm = test_norm /  1.0
        # return normalized images
    return train_norm, test_norm

# prepare pixel data
trainX, testX = prep_pixels(trainX, testX, resnet_mode=parameters['resnet_mode'])

def args_to_dict(**kwargs):
    return kwargs
position_dim = (int(parameters['max_length']),int(parameters['res']),int(parameters['res']),2) if  parameters['broadcast']==1 else (int(parameters['max_length']),2)
movie_dim = (int(parameters['max_length']), int(parameters['res']), int(parameters['res']), 3)
    
generator_params = args_to_dict(batch_size=64, 
                                    movie_dim=movie_dim, 
                                    position_dim=position_dim, 
                                    n_classes=None, 
                                    shuffle=True,
                                    prep_data_per_batch=True,
                                    one_hot_labels=False, 
                                    one_random_sample=False,
                                    res = int(parameters['res']),
                                    n_samples = int(parameters['n_samples']),
                                    mixed_state = True,
                                    n_trajectories = int(parameters['trajectories_num']),
                                    trajectory_list = 0,
                                    broadcast=parameters['broadcast'],
                                    style = parameters['style'],
                                    max_length=int(parameters['max_length']),
                                    noise = parameters['noise'],
                                    time_sec=parameters['time_sec'], 
                                    traj_out_scale=parameters['traj_out_scale'],  
                                    snellen=parameters['snellen'],
                                    vm_kappa=parameters['vm_kappa'],
                                    random_n_samples = 0,
                                    )
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
    val_generator_classifier = Syclopic_dataset_generator(testX, testY, **generator_params)
    train_generator_classifier = Syclopic_dataset_generator(
                            trainX[:-5000], 
                            trainY[:-5000], 
                            **generator_params)
# train_generator_features = Syclopic_dataset_generator(trainX[:-5000], None, teacher=fe_model, **generator_params)
# val_generator_features = Syclopic_dataset_generator(trainX[-5000:].repeat(parameters['val_set_mult'],axis=0), None, teacher=fe_model, validation_mode=True, **generator_params)

#%%

#print('The Decoder Baseline Accuracy = ', parameters['decoder_max_test_error'])
#########################       Start Evaluation      ################################
################################################################################
######################### Combine Student and Decoder ##########################
################################################################################
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
s_to_w_t = np.array(scatter_data['t']).argsort()[::-1]
s_to_w_s = np.array(scatter_data['s']).argsort()[::-1]
s_to_w_st = np.array(scatter_data['st']).argsort()[::-1]
print('Top Spatio-Temporal features - ', s_to_w_st[:10])
print('Top Spatial features - ', s_to_w_s[:10])
print('Top Temporal features - ', s_to_w_t[:10])
#%%
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
after_zero_dataframe.to_pickle(dump_folder + '/after_zero_dataframe_{}'.format(run_name), protocol=3)
#%%
########################     Evaluating 1 to 10        ######################### 
########################     strongest features        #########################
########################        of t, s & st           #########################
all_ratings = [s_to_w_st, s_to_w_s, s_to_w_t, np.random.randint]
all_names = ['tempo_spatial','spatial','temporal', 'random']
silance_slice = pd.DataFrame(columns = ['num_silanced','type','accur','delta_accur'])
#Going one by one - st, s, t - and silancing the 5 - 10 - 20 most activated features.
full_accur = student_and_decoder.evaluate(val_generator_classifier, verbose = 2)[1]
for top in [1,2,3,5,7,10]:
    temp_row = []
    print('####################### ', top, ' ################################')
    for indx, mode in enumerate(all_ratings):
        if indx == 3:
            rng = np.random.default_rng()
            mode = rng.choice(64, 64, replace = False)
        model_copy = keras.models.clone_model(student)
        model_copy.set_weights(student.get_weights()) 
        for k in range(top):
            model_copy = silence_feature( model_copy = model_copy,
                                     orig_model = model_copy, 
                                 index = mode[k])
        
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
        print('--------- ',all_names[indx],' ----------')
        accur = student_and_decoder.evaluate(val_generator_classifier, verbose = 2)[1]
        temp_row = {'num_silanced':top, 'type':all_names[indx], 'accur':accur, 'delta_accur':full_accur - accur}
        print('When silancing the top {} in {} we get a {} decrease in accuracy'.format(top, all_names[indx], full_accur - accur))
        silance_slice = silance_slice.append(temp_row, ignore_index = True)
silance_slice.to_pickle(dump_folder + '/silance_slice_{}'.format(run_name), protocol=3)
plt.figure()
sns.barplot(data=silance_slice, x='num_silanced', y="delta_accur", hue="type")
plt.savefig(dump_folder + '/accuracy_drop_by_features.png')
#%%
plt.figure()
sns.lineplot(data = silance_slice, x = 'num_silanced', y = 'delta_accur', hue = 'type', marker = 'o')
plt.savefig(dump_folder + '/accuracy_drop_lineplot.png')