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
print(os.getcwd() + '/')


#%%
############################### Get Trained Student ##########################3
run_name = 'noname_j42560_t1632392515'
student, parametrs = load_student(run_name = run_name)


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
def define_generator(latent_dim = 100):
    input = keras.layers.Input(shape=[latent_dim])
    initial_shape = [1,1,1,128]
    
    final_shape = [10,8,8,3]
    x = keras.layers.Dense(initial_shape[0]*initial_shape[1]*initial_shape[2]*initial_shape[3],
                           )(input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Reshape(initial_shape)(x)
    #print(x.shape)
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
generator = define_generator_temporal()
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
    iterations = 60
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

# The dimensions of our input image
img_width = 8
img_height = 8
loss_list2 = []

runs = 5
temp_image = np.ones(shape = [10,runs,3])
for i in range(runs):
    print(i)
    feature = 50
    #regular
    generator = define_generator_temporal(latent_dim = 100)
    optimizer = tf.keras.optimizers.Adam(lr=5e-4)
    loss, img = visualize_filter(feature)
    loss_list2.append(-np.array(loss))
    
    for j in range(10):
        color = img[j,0,0,:]
        temp_image[j,i,:] *= color
        


plt.figure()
plt.imshow(temp_image)
plt.figure()
plt.imshow(temp_image.mean(axis=-1), cmap='gray')

plt.figure()
j=0
for i in loss_list2:
    plt.plot(i, label = j)
    j+=1
plt.legend()
#%%
loss_list_spatial = []
for run in range(runs):
    #spatial
    generator = define_generator_spatial()
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    loss, img = visualize_filter(feature)
    img_list_spatial[feature] = img
    loss_list_spatial.append(-np.array(loss))


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
for i in range(64):
    print('#############    ',i,'    ##################')
    feature = i
    feature_list.append(feature)
    #regular
    generator = define_generator(latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    loss, img = visualize_filter(feature)
    img_list[feature] = img
    #img_list.append(img)
    loss_list.append(-np.array(loss))
    
    fig, ax = plt.subplots(3,3)
    fig.suptitle(feature)
    indx = 0
    for l in range(3):
        for k in range(3):
            ax[l,k].imshow(img[indx,:,:,:])
            indx+=1
            ax[l,k].title.set_text(indx)
    #now, shuffle and test loss
    img_sh = tf.random.shuffle(img)

    img_sh = np.expand_dims(img_sh, 0)
    sh_loss = compute_loss(img_sh,feature)
    #Add noise 
    img_ns = img + tf.random.normal(img.shape,stddev = np.std(img))
    img_ns = np.expand_dims(img_ns, 0)
    ns_loss = compute_loss(img_ns,feature)
    shuffle.append([-float(loss[-1]),-float(sh_loss),-float(ns_loss),])
    ##########################################################################
    #spatial
    generator = define_generator_spatial(latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    loss, img = visualize_filter(feature)
    img_list_spatial[feature] = img
    loss_list_spatial.append(-np.array(loss))
    fig, ax = plt.subplots(3,3)
    fig.suptitle(feature)
    indx = 0
    for l in range(3):
        for k in range(3):
            ax[l,k].imshow(img[indx,:,:,:])
            indx+=1
            ax[l,k].title.set_text(indx)
    #now, shuffle and test loss
    img_sh = tf.random.shuffle(img)

    img_sh = np.expand_dims(img_sh, 0)
    sh_loss = compute_loss(img_sh,feature)
    #Add noise 
    img_ns = img + tf.random.normal(img.shape,stddev = np.std(img))
    img_ns = np.expand_dims(img_ns, 0)
    ns_loss = compute_loss(img_ns,feature)
    spatial_shuffle.append([-float(loss[-1]),-float(sh_loss),-float(ns_loss),])
    ##########################################################################
    #temporal
    best_loss = None
    for run in range(1):
        generator = define_generator_temporal(latent_dim=latent_dim)
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        loss, img = visualize_filter(feature)
        img_list_temporal[feature] = img
        loss_list_temporal.append(-np.array(loss))
    fig, ax = plt.subplots(3,3)
    fig.suptitle(feature)
    indx = 0
    for l in range(3):
        for k in range(3):
            ax[l,k].imshow(img[indx,:,:,:])
            indx+=1
            ax[l,k].title.set_text(indx)
    #now, shuffle and test loss
    img_sh = tf.random.shuffle(img)

    img_sh = np.expand_dims(img_sh, 0)
    sh_loss = compute_loss(img_sh,feature)
    #Add noise 
    img_ns = img + tf.random.normal(img.shape,stddev = np.std(img))
    img_ns = np.expand_dims(img_ns, 0)
    ns_loss = compute_loss(img_ns,feature)
    temporal_shuffle.append([-float(loss[-1]),-float(sh_loss),-float(ns_loss),])
    
pickle.dump(img_list,open('img_vis_{}'.format(run_name),'wb'))
pickle.dump(img_list_spatial,open('img_spatial_{}'.format(run_name),'wb'))
pickle.dump(img_list_temporal,open('img_temporal_{}'.format(run_name),'wb'))

loss_t = np.array(loss_list_temporal)[:,-1]
loss_s = np.array(loss_list_spatial)[:,-1]
loss_st = np.array(loss_list)[:,-1]

pickle.dump(loss_st,open('loss_st_{}'.format(run_name),'wb'))
pickle.dump(loss_s,open('loss_s_{}'.format(run_name),'wb'))
pickle.dump(loss_t,open('loss_t_{}'.format(run_name),'wb'))

plt.figure()
plt.plot(loss_t,loss_s,'o')
plt.xlabel('Purely Temporal Activation')
plt.ylabel('Purly Spatial Activation')

sns.scatterplot(data=data, x="t", y="s", hue="st", size="st")
plt.legend(bbox_to_anchor=(-0.1, 1), loc=0, borderaxespad=0.)
plt.xlabel('Purely Temporal Activation')
plt.ylabel('Purly Spatial Activation')


########### 10X3
feature = np.array(loss_s).argsort()[-1:][::-1]
img_l = []
img_l.append(img_list[feature])
img_l.append(img_list_spatial[feature])
img_l.append(img_list_temporal[feature])
fig, ax = plt.subplots(3,10)
indx = 0
for k in range(3):
    for l in range(10):
        ax[k,l].axis('off')
        ax[k,l].imshow(img_l[k][l,:,:,:])
fig.suptitle('feature = {}'.format(feature),y=0.52)
plt.subplots_adjust(left=0.125,
                    bottom=0.0, 
                    right=0.9, 
                    top=0.5, 
                    wspace=0.2, 
                    hspace=0.0)

########### 5*3
feature = 24
img_l = []
img_l.append(img_list[feature])
img_l.append(img_list_spatial[feature])
img_l.append(img_list_temporal[feature])
fig, ax = plt.subplots(3,5)
indx = 0
for k in range(3):
    for l in range(5):
        ax[k,l].axis('off')
        ax[k,l].imshow(img_l[k][l,:,:,:])
fig.suptitle('feature = {}'.format(feature),y=0.58)
plt.subplots_adjust(left=0.125,
                    bottom=0.0, 
                    right=0.9, 
                    top=0.5, 
                    wspace=0.2, 
                    hspace=0.1)

#%%
#generate temporal friquancy time series
def generate_frequancy_ts(frequancy = 5, phase = 0, samples = 10, res = 8, 
                          grayscale = False, 
                          color_code = [50,50,50]):
    
    #frequancy is defined as the number of picks in the ts
    #phase controls the location of the picks, phase=0 indicates that on t=0
    #the images are on min_val
    every_how_much_samples = 10//frequancy
    if frequancy == 1:
        freq_vec = (np.sin(np.linspace(1.5*np.pi, 2*np.pi+1.5*np.pi, 10))+1)/2
    elif frequancy == 2:
        freq_vec = np.array([0.  , 0.41317591, 0.96984631, 0.6      , 0.05,
       0.3, 0.75  , 0.96984631, 0.41317591, 0. ])     
    elif frequancy == 3:
        freq_vec = np.array([0.  , 1, 0.6, 0.  , 0.95, 0.4, 0.  , 0.6, 1, 0.  ])
    elif frequancy == 4:
        freq_vec = np.array([0.        , 0.96984631, 0.11697778, 1.      , 0.41317591,
       0., 0.99      , 0.011697778, 0.96984631, 0.  ])
    elif frequancy == 5:
        freq_vec = (np.sin(np.linspace(1.5*np.pi, 10*np.pi+0.5*np.pi, 10))+1)/2
        
    if phase:
        freq_vec = np.roll(freq_vec, phase)
    
    img_ts = np.ones([samples, res,res, 3])
    for indx, freq in enumerate(freq_vec):
        if grayscale:    
            img_ts[indx] = np.ones([1, res,res, 3])*freq*color_code[0]
        else:
            img_ts[indx] = np.ones([1, res,res, 3])*freq*(np.array(color_code))
    img_ts /= 255.
    return img_ts

#%%
#Check activation of feature given a frequancy/color_code/phase 

#Starting with grayscale:

freq_list = [1,2,3,4,5]
phase_list = [1,3,4,5,6,7,8,9,10]

traject = np.array(create_trajectory((32,32),sample = 10, style = 'spiral'))#tf.random.uniform((10,2))
traject = np.expand_dims(traject, 0)
cc_list = []
for r in [255.,0.]:
    for g in [255.,0.]:
        for b in [255.,0.]:
           cc_list.append([r,g,b])
freq_loss_dict_list = []
freq_parametrs = pd.DataFrame()
colorwise_dict = {}
for feature in range(64
                     ):
    print('')
    feature_params = {}
    freq_loss_list = []
    freq_loss_dict = {}
    feature = feature
    
    
    #print the temporal loss found by the generator
    img = img_list_temporal[feature]
    img = np.expand_dims(img,0)
    activation = feature_extractor((img,traject))
    filter_activation = activation[0][:, 1:-1, 1:-1, feature]
    old_loss = tf.reduce_mean(filter_activation)
    print('#######################################')
    print('feature: ', feature )
    print(loss_st[feature], loss_s[feature],loss_t[feature],)
    feature_params['feature'] = feature
    feature_params['loss_st'] = loss_st[feature]
    feature_params['loss_s'] = loss_s[feature]
    feature_params['loss_t'] = loss_t[feature]
    color_loss = []
    for color_code in cc_list:
        temp_color_loss = []
        for phase in phase_list:
            phase_loss_list = []
            for frequency in freq_list:
            
            
                
                samples = 10
                res = 8
                grayscale = False
                img = generate_frequancy_ts(frequency, phase, samples, res, 
                                          grayscale, 
                                          color_code)
                img = np.expand_dims(img,0)
                activation = feature_extractor((img,traject))
                filter_activation = activation[0][:, 1:-1, 1:-1, feature]
                freq_loss_list.append(tf.reduce_mean(filter_activation))
                phase_loss_list.append(tf.reduce_mean(filter_activation))
                freq_loss_dict[frequency,phase,tuple(color_code)] = tf.reduce_mean(filter_activation)

            temp_color_loss.append(phase_loss_list)
        color_loss.append(temp_color_loss)
    plt.subplots(2,4,figsize=(12,7))
    indx = 1
    for cll in color_loss:
        plt.subplot(2,4,indx)
        for pll in cll:
            p = pll
            plt.plot(pll,'o')
        plt.title(cc_list[indx-1])
        indx+=1
    plt.subplots_adjust(left=0.125,
                        bottom=0.0, 
                        right=0.9, 
                        top=0.5, 
                        wspace=0.4, 
                        hspace=0.7)
    freq_loss_dict_list.append(freq_loss_dict)
    colorwise_dict[feature] = color_loss
    plt.figure()
    plt.plot(freq_loss_list, 'o')
    plt.title('feature = {}, generator loss = \n{} {} {}'.format(feature, loss_st[feature],loss_s[feature],loss_t[feature]))
    print(np.max(freq_loss_list), np.min(freq_loss_list),np.mean(freq_loss_list))
    max_params = []
    for max_indx in np.array(freq_loss_list).argsort()[-3:][::-1]:
        max_params.append(list(freq_loss_dict.keys())[max_indx])
    max_vals = np.array(freq_loss_list)[np.array(freq_loss_list).argsort()[-3:][::-1]]
    print(max_vals)
    print(max_params)
    for num in range(3):
        feature_params['max{}'.format(num)] = max_vals[num]
        feature_params['max{}_freq'.format(num)] = max_params[num][0]
        feature_params['max{}_phase'.format(num)] = max_params[num][1]
        feature_params['max{}_color'.format(num)] = max_params[num][2]
    min_params = []
    for min_indx in np.array(freq_loss_list).argsort()[:3]:
        min_params.append(list(freq_loss_dict.keys())[min_indx])
    min_vals = np.array(freq_loss_list)[np.array(freq_loss_list).argsort()[:3]]
    print(min_vals)
    print(min_params)
    for num in range(3):
        feature_params['min{}'.format(num)] = min_vals[num]
        feature_params['min{}_freq'.format(num)] = min_params[num][0]
        feature_params['min{}_phase'.format(num)] = min_params[num][1]
        feature_params['min{}_color'.format(num)] = min_params[num][2]
    freq_parametrs = freq_parametrs.append(feature_params, ignore_index = True)

freq_parametrs = freq_parametrs[['feature', 'loss_st', 'loss_s', 'loss_t', 'max0', 'max0_freq', 'max0_phase', 'max0_color', 'max1', 'max1_freq', 'max1_phase', 'max1_color', 'max2', 'max2_freq', 'max2_phase', 'max2_color', 'min0', 'min0_freq', 'min0_phase', 'min0_color', 'min1', 'min1_freq', 'min1_phase', 'min1_color', 'min2', 'min2_freq', 'min2_phase', 'min2_color']]
#freq_parametrs.to_pickle('freq_parametrs_{}'.format(run_name))

# fig, ax = plt.subplots(3,3)
# indx = 0
# for l in range(3):
#     for k in range(3):
#         ax[l,k].imshow(img[indx,:,:,:])
#         indx+=1
#         ax[l,k].title.set_text(indx)
        
#%%
mod_fac = []
for i in range(3):
    new_img = img_vis[i].reshape([10,-1])
    global_mean = new_img.mean()
    centered_img = new_img - global_mean
    pxl_mean_time = np.sqrt( np.mean(centered_img.mean(axis=0)**2))
    pxl_std_time = np.sqrt( np.mean(centered_img.var(axis=0)))
    modulation_index1 = pxl_std_time / pxl_mean_time
    mod_fac.append(modulation_index1)
    # mean = new_img.mean(axis = 0)
    # sum_ = new_img.sum(axis = 0)
    # new_img -= mean

    
    # corr = new_img @ new_img.T
    # val, vec = np.linalg.eig(corr)
    
    # plt.figure()
    # plt.plot(vec[0])
    # plt.plot(vec[1])

        
#%%
# plt.figure()
# for idx, l in enumerate(loss_list):
#     plt.plot(l, label = feature_list[idx])
# plt.legend()
# plt.title('regular')
# plt.figure()
# for idx, l in enumerate(loss_list_spatial):
#     plt.plot(l, label = feature_list[idx])
# plt.legend()
# plt.title('spatial')
# plt.figure()
# for idx, l in enumerate(loss_list_temporal):
#     plt.plot(l, label = feature_list[idx])
# plt.legend()
# plt.title('temporal')

plt.figure()
#keras.preprocessing.image.save_img("0.png", img)
spatial_shuffle = np.array(spatial_shuffle)
temporal_shuffle = np.array(temporal_shuffle)
plt.plot(temporal_shuffle[:,0], spatial_shuffle[:,0], 'o')
plt.xlabel('temporal only')
plt.ylabel('spacial only')
plt.ylim(0,0.3)
plt.xlim(0,0.4)


#display(Image("0.png"))
#%%
img_list_ = []
for i in range(10):
    print('#############    ',i,'    ##################')
    feature = 23
    feature_list.append(feature)
    #regular
    generator = define_generator(latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    loss, img = visualize_filter(feature)
    img_list_.append(img)
    #img_list.append(img)
    loss_list.append(-np.array(loss))
    
    # fig, ax = plt.subplots(3,3)
    # fig.suptitle(feature)
    # indx = 0
    # for l in range(3):
    #     for k in range(3):
    #         ax[l,k].imshow(img[indx,:,:,:])
    #         indx+=1
    #         ax[l,k].title.set_text(indx)  
#%%
for img in img_list_:
    pca = PCA(n_components=3)
    pca_input = img.reshape(10,8*8*3)
    student_pca = pca.fit_transform(pca_input)
    print(pca.explained_variance_ratio_)
    b_val = 0.1
    g_val = 0.9
    #plot first component
    plt.figure()
    plt.title(feature)
    plt.plot(student_pca[:,0])
    plt.figure()
    plt.title(feature)
    for i in range(10):
        plt.plot(student_pca[i,0],student_pca[i,1],'o', color = (0.01, g_val, b_val))
        b_val+=0.1
        g_val-=0.1
    student_mean_vec = student_pca.mean(axis = 0)
    plt.plot(student_mean_vec[0], student_mean_vec[1], 'o', color = (1,0,0))
    plt.plot(student_pca[:,0],student_pca[:,1])
    

