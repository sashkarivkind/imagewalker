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
import scipy.stats as stats
import pandas as pd
import time
import pickle
import argparse
from feature_learning_utils import  load_student, student3

from keras_utils import create_trajectory
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


### teacher network parameters
parser.add_argument('--teacher_net', default='/home/orram/Documents/GitHub/imagewalker/teacher_student/model_510046__1628691784.hdf', type=str, help='path to pretrained teacher net')

parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
parser.add_argument('--last_layer_size', default=128, type=int, help='last_layer_size')


parser.add_argument('--dropout1', default=0.2, type=float, help='dropout1')
parser.add_argument('--dropout2', default=0.0, type=float, help='dropout2')
parser.add_argument('--dataset_norm', default=128.0, type=float, help='dropout2')
parser.add_argument('--dataset_center', dest='dataset_center', action='store_true')
parser.add_argument('--no-dataset_center', dest='dataset_center', action='store_false')


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

parser.set_defaults(data_augmentation=True,layer_norm_res=True,layer_norm_student=True,layer_norm_2=True,skip_conn=True,last_maxpool_en=True, testmode=False,dataset_center=True)

config = parser.parse_args()
config = vars(config)
print('config  ',config)

parameters = config
TESTMODE = parameters['testmode']


lsbjob = os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob

this_run_name = parameters['run_name_prefix'] + '_j' + lsbjob + '_t' + str(int(time.time()))
parameters['this_run_name'] = this_run_name


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
############################### Get Trained Student ##########################3
student, parametrs = load_student(run_name = 'noname_j253106_t1631441888')


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

def compute_loss(input_image, filter_index):
    
    activation = feature_extractor(input_image)
    
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 1:-1, 1:-1, filter_index]
    
    return tf.reduce_mean(filter_activation)



def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        traject = np.array(create_trajectory((32,32),sample = 10, style = 'spiral'))#tf.random.uniform((10,2))
        traject = tf.expand_dims(tf.convert_to_tensor(traject), axis = 0)
        tape.watch(img)
        loss = compute_loss((img,traject), filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    orig_img = img + 0
    orig_img += learning_rate * grads
    return loss, orig_img


def initialize_image(broadcast = False):
    # We start from a gray image with some random noise
    img = tf.random.uniform((1,10, img_width, img_height, 3))
    traject = np.array(create_trajectory((32,32),sample = 10, style = 'spiral'))#tf.random.uniform((10,2))
    if broadcast:
        broadcast_place = np.ones(shape = [10,img_width,img_height,2])
        for i in range(10):
            broadcast_place[i,:,:,0] *= traject[i,0]
            broadcast_place[i,:,:,1] *= traject[i,1]
    else:
        broadcast_place = traject
        
    broadcast_place = tf.expand_dims(tf.convert_to_tensor(broadcast_place), axis = 0)
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25# broadcast_place)


def visualize_filter(filter_index, use_img = False):
    # We run gradient ascent for 20 steps
    iterations = 100
    learning_rate = 0.2
    if use_img:
        img = tf.expand_dims(tf.convert_to_tensor(images[42]), axis = 0)/255
    else:
        img = initialize_image()
    loss_list = []
    for iteration in range(iterations):
        temp_img = keras.layers.LayerNormalization(axis = [1,2])(img)
        img = temp_img
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)
        loss_list.append(loss)

    # Decode the resulting input image
    img = deprocess_image(img.numpy())
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

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img
#%%
from IPython.display import Image, display
import matplotlib.pyplot as plt 
# The dimensions of our input image
img_width = 8
img_height = 8
loss_list = []
for i in range(1):
    
    loss, img = visualize_filter(20)
    loss_list.append(loss)
    fig, ax = plt.subplots(3,3)
    indx = 0
    for l in range(3):
        for k in range(3):
            ax[l,k].imshow(img[0,indx,:,:,:])
            indx+=1
            ax[l,k].title.set_text(indx)

#keras.preprocessing.image.save_img("0.png", img)

#display(Image("0.png"))
#%%
# Compute image inputs that maximize per-filter activations
# for the first 64 filters of our target layer
all_imgs = []
for filter_index in range(64):
    print("Processing filter %d" % (filter_index,))
    loss, img = visualize_filter(filter_index, use_img=True)
    all_imgs.append(img)
#%%
# Build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 1
n = 8
cropped_width = img_width 
cropped_height = img_height
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((64*4, 64*4, 3))

# Fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img = all_imgs[i * n + j]
        stitched_filters[
            (32) * i : (32) * i + 32,
            (32) * j : 32 * j+ 32,
            :,
        ] = img
keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)

from IPython.display import Image, display

display(Image("stiched_filters.png"))

#%%

