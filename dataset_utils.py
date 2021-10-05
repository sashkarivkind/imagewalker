#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some order to the code - the utils!
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import cv2
import misc
from RL_networks import Stand_alone_net
import pandas as pd
import random
import os 

# import importlib
# importlib.reload(misc)


import matplotlib.pyplot as plt
import SYCLOP_env as syc

import tensorflow as tf
import tensorflow.keras as keras
from drift_intoy_and_rucci20 import gen_drift_traj_condition

import numpy as np

def build_cifar_padded(image,pad_size = 100, xx=132,yy=132,y_size=32,x_size=32,offset=(0,0)):
    #todo: double-check x-y vs. row-column convention
    #prepares an mnist image padded with zeros everywhere around it, written in a somewhat strange way to resuse other availiable functions
    
    image = cv2.copyMakeBorder( image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
    return image

def vonmises_walk(vm_bias=np.sqrt(2.), vm_amp=1., kappa=0, n_steps=5, enforce_origin=True):
    phi0 = 2 * np.pi * np.random.uniform()
    flip = 0
    if kappa < 0:
        kappa = -kappa
        flip = np.pi
    dphi = flip + np.random.vonmises(0, kappa, size=n_steps)
    dr = vm_bias + vm_amp * np.abs(np.random.normal(size=n_steps))
    if enforce_origin:
        dr[0] = 0
    phi = phi0 + np.cumsum(dphi)
    dxy = dr * np.array([np.cos(phi), np.sin(phi)])
    xy = np.cumsum(dxy.T, axis=0)
    return xy


def test_num_of_trajectories(gen,batch_size=32):
    '''test how many actual trajectories are there in batch'''
    zz=[]
    cc=0
    for uu in range(len(gen)):
        for bb in range(batch_size):
            zz.append(str(gen[uu][0][1][bb, :, 0, 0, :]))
            cc += 1
    return len(set(zz)), cc

#Define function for low resolution lens on syclop
def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp

def bad_res102(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    return dwnsmp

def create_trajectory(starting_point, n_samples = 5, style = 'brownian', noise = 0.15, time_sec=0.3, traj_out_scale=None,  snellen=True, vm_kappa=0):
    if style[:3] == 'xx1':
        if style == 'xx1_intoy_rucci':
            _, steps = gen_drift_traj_condition(duration=time_sec, N=n_samples, snellen=snellen)
            steps = starting_point+traj_out_scale*steps.transpose()
        if style == 'xx1_vonmises_walk':
            steps = vonmises_walk(n_steps=n_samples, kappa=vm_kappa)
            steps += starting_point
    else:
        steps = []

        phi = np.random.randint(0.1,2*np.pi) #the angle in polar coordinates

        ##bug fixes
        if style == 'degenerate_fix2':
            phi = np.random.random()*2*np.pi #(0.1, 2 * np.pi)  # the angle in polar coordinates
            style = 'degenerate_fix'

        if style == 'spiral_2dir2':
            phi = np.random.random() * 2 * np.pi  # (0.1, 2 * np.pi)  # the angle in polar coordinates
            style = 'spiral_2dir'
        ## end bug fixes

        speed = 0.8#np.abs(0.5 + np.random.normal(0,0.5))         #Constant added to the radios
        r = 3
        name_list = ['const direction + noise','ZigZag','spiral', 'brownian','degenerate']
        speed_noise = speed * 0.2
        phi_noise = 0.05
        x, y = starting_point[1], starting_point[0]
        steps.append([y,x])
        phi_speed =  (1/8)*np.pi
        old_style = style
        for j in range(n_samples-1):
            style = old_style
            if style == 'mix':
                old_style = 'mix'
                style = random.sample(name_list, 1)
            if style == 'const_p_noise':
                r += speed + np.random.normal(-0.5,speed_noise)
                phi_noise = noise
                phi_speed = np.random.normal(0,phi_noise)
                phi += phi_speed
            elif style == 'ZigZag':
                r += speed + np.random.normal(-0.5,speed_noise)
                phi_noise = 0.005
                phi_speed *=  -1
                phi += phi_speed + np.random.normal(0,phi_noise)
            elif style == 'spiral':
                r += speed / 2 + np.random.normal(-0.5, speed_noise)
                phi_noise = 0.1
                phi_speed = np.random.normal((2 / 4) * np.pi, (1 / 8) * np.pi)
                factor = 1  # np.random.choice([-1,1])
                phi += factor * phi_speed
            elif style == 'spiral_2dir' or style == 'spiral_2dir_shfl':
                r += speed / 2 + np.random.normal(-0.5, speed_noise)
                phi_noise = 0.1
                phi_speed = np.random.normal((2 / 4) * np.pi, (1 / 8) * np.pi)
                if j==0:
                    factor = np.random.choice([-1,1])
                phi += factor * phi_speed
            elif style == 'big_steps':
                r += speed/2 + np.random.normal(-0.5,speed_noise)
                phi_noise = 0.1
                phi_speed = np.random.normal((2/4)*np.pi,(1/8)*np.pi)
                factor = np.random.choice([-1,1])
                phi += factor * phi_speed
            elif style == 'brownian':
                r += speed/2 + np.random.normal(-0.5,speed_noise)
                phi = np.random.randint(0.1,2*np.pi)
            elif style == 'degenerate' or style == 'degenerate_fix':
                r += speed + np.random.normal(-0.5,speed_noise)

            elif style == 'old':

                starting_point += np.random.randint(-2,3,2)
                r = 0
                phi = 0
            else:
                error

            x, y = starting_point[1] + int(r * np.cos(phi)), starting_point[0]+int(r * np.sin(phi))
            if style == 'degenerate_fix':
                while abs(steps[-1][0]-y)<1e-6 and abs(steps[-1][1]-x)<1e-6:
                    r += 0.5
                    x, y = starting_point[1] + int(r * np.cos(phi)), starting_point[0] + int(r * np.sin(phi))

            steps.append([y,x])

            #shuffling all the trajectory except for the firs point if appropriate flag on
            if style == 'spiral_shfl' or style == 'spiral_2dir_shfl':
                step0 = steps[0]
                steps_ = steps[1:]
                random.shuffle(steps_)
                steps = [step0] + steps_

    return steps


def generate_syclopic_images(images, res, n_samples = 5, mixed_state = True, add_traject = True,
                   trajectory_list=0, n_trajectories = 20,
                   bad_res_func = bad_res102, up_sample = False, broadcast = 0,
                   style = 'brownian', noise = 0.15, max_length = 20, loud=False, **kwargs):
    '''
    Creates a keras dataloader object of syclop outputs
    from a list of images and labels.
    
    Parameters
    ----------
    images : List object holding the images to proces
    labels : List object holding the labels
    res : resolution dawnsampling factor - to be used in cv.resize(orig_img, res)
    n_samples: the number of samples to have in syclop
    mixed_state : if False, use the same trajectory on every image.
    return_datasets: rerutns datasets rather than dataloaders
    Returns
    -------
    train_dataloader, test_dataloader - torch DataLoader class objects

    '''
    def extended_trajectory_builder():
        #Setting the coordinates to visit
        if type(trajectory_list) is int:
            if trajectory_list:
                np.random.seed(trajectory_list)
            starting_point = np.array([agent.max_q[0]//2,agent.max_q[1]//2])
            steps = create_trajectory(starting_point= starting_point,
                                      n_samples = n_samples,
                                      style = style,
                                       noise = noise,**kwargs)

            if mixed_state and n_trajectories!= -1:
                seed_list.append(new_seed)
                q_sequence = np.array(steps).astype(int)
            else:
                if count == 0:
                    q_sequence = np.array(steps).astype(int)
        else:
            q_sequence = np.array(trajectory_list[img_num]).astype(int)

        return q_sequence
    '''end of auxillary function'''
    if n_samples > max_length:
        max_length = n_samples
        print('max_length ({}) must be >= n_samples ({}), changed max_length to be == n_samples'.format(max_length, n_samples))
    count = 0
    ts_images = []
    dvs_images = []
    q_seq = []
    seed_list = []
    count = 0


    if mixed_state and n_trajectories!= -1 :
        np.random.seed(42)
        new_seed = 42
        
    #initiating syclop instance for generating sequences of images
    img = build_cifar_padded(1. / 256 * images[0])
    scene = syc.Scene(image_matrix=img)
    if up_sample:
        sensor = syc.Sensor(winx=56, winy=56, centralwinx=32, centralwiny=32, nchannels=3,
                            resolution_fun=lambda x: bad_res_func(x, (res, res)), resolution_fun_type='down')
    else:
        sensor = syc.Sensor(winx=32, winy=32, centralwinx=res // 2, centralwiny=res // 2, nchannels=3,
                            resolution_fun=lambda x: bad_res102(x, (res, res)), resolution_fun_type='down')
    agent = syc.Agent(max_q=[scene.maxx - sensor.hp.winx, scene.maxy - sensor.hp.winy])

    for img_num,img in enumerate(images):
        ##set a random seed in the range specified by the number of trajectories
        if n_trajectories > 0:
            new_seed = random.randint(0,n_trajectories)
            np.random.seed(new_seed)    
        orig_img = img*1
        if img_num == 42 and loud:
            print('Are we Random?? ', np.random.randint(1, 20))

        #Set the padded image
        img=build_cifar_padded(1./256*img)
        img_size = img.shape

        if n_trajectories == -1:
            starting_point = np.array([agent.max_q[0]//2,agent.max_q[1]//2])
            steps = create_trajectory(starting_point= starting_point,
                                      n_samples = n_samples,
                                      style = style,
                                       noise = noise,**kwargs)
            q_sequence = np.array(steps).astype(int)
        else:
            q_sequence = extended_trajectory_builder()

        if count == 0 and loud:
            print(q_sequence.shape)
            
        #Create empty lists to store the syclops outputs
        imim=[]
        scene = syc.Scene(image_matrix=img)
        agent.set_manual_trajectory(manual_q_sequence=q_sequence)
        for t in range(len(q_sequence)):
            agent.manual_act()
            sensor.update(scene, agent)
            imim.append(sensor.frame_view)
        imim = np.array(imim)

        #Add current proccessed image to lists
        ts_images.append(imim)
        if broadcast==1:
            broadcast_place = np.ones(shape = [n_samples,res,res,2])
            for i in range(n_samples):
                broadcast_place[i,:,:,0] *= q_sequence[i,0]
                broadcast_place[i,:,:,1] *= q_sequence[i,1]
            q_seq.append(broadcast_place/img_size[0])
        else:
            q_seq.append(q_sequence/img_size[0])
        count += 1
    if loud:
        print(q_sequence)
    #pre pad all images to max_length
    for idx, image in enumerate(ts_images):
        image_base = np.zeros(shape = [max_length, res, res, 3])
        if broadcast==1:
            seq_base = np.zeros(shape = [max_length, res, res, 2])
        else:
            seq_base = np.zeros([max_length, 2])
        image_base[-len(imim):] = image
        seq_base[-len(q_sequence):] = q_seq[idx]
        ts_images[idx] = image_base * 1
        q_seq[idx] = seq_base * 1

    if add_traject:
        return (np.array(ts_images),np.array(q_seq))
    else:
        return np.array(ts_images)

class Syclopic_dataset_generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, labels, batch_size=None, movie_dim=None, position_dim=None,
                 n_classes=None, shuffle=True, syclopic_function=generate_syclopic_images, retutn_x0_only=False,
                 prep_data_per_batch=False,one_hot_labels=False, one_random_sample=False, validation_mode=False, loud_en=False, teacher=None, preprocess_fun=lambda x:x, augmenter=None, **kwargs):
        list_IDs = list(range(len(images)))
        'Initialization'
        self.images=images
        self.movie_dim = movie_dim
        self.position_dim = position_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.syclopic_function = syclopic_function
        self.prep_data_per_batch=prep_data_per_batch
        self.kwargs = kwargs
        self.one_hot_labels = one_hot_labels
        self.one_random_sample = one_random_sample
        self.validation_mode = validation_mode
        self.loud_en = loud_en
        self.teacher = teacher
        self.preprocess_fun = preprocess_fun
        self.retutn_x0_only = retutn_x0_only
        if validation_mode:
            self.prep_data_per_batch = False
            if prep_data_per_batch:
                print('overriding prep_data_per_batch for validation mode!!')
        self.on_standard_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        if self.validation_mode:
            self.on_validation_epoch_end()
        else:
            self.on_standard_epoch_end()

    def on_validation_epoch_end(self):
        pass

    def on_standard_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        if not self.prep_data_per_batch:
            self.full_syclopic_view = self.syclopic_function(self.images,loud=self.loud_en,**self.kwargs)

        self.loud = self.loud_en

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # Generate data
        if self.prep_data_per_batch:
            X = self.syclopic_function(self.images[list_IDs_temp],loud=self.loud,**self.kwargs)
            if self.teacher is None:
                y = self.labels[list_IDs_temp]
            self.loud = False
        else:
            X1 = self.full_syclopic_view[0][list_IDs_temp]
            X2 = self.full_syclopic_view[1][list_IDs_temp]
            if self.teacher is None:
                y = self.labels[list_IDs_temp]
            X=(X1,X2)
        X = (self.preprocess_fun(X[0]), X[1])

        if self.teacher is None:
            y = keras.utils.to_categorical(y, num_classes=self.n_classes) if self.one_hot_labels else y
        else:
            y = self.teacher.predict(self.images[list_IDs_temp])

        if self.one_random_sample:  # returning  one random sample from each sequence (used in baseline tests)
            data_len = np.shape(X[0])[0]
            indx = list(range(data_len))
            pick_sample = np.random.randint(self.kwargs['n_samples'], size=data_len)
            return X[0][indx,pick_sample,...], y
        else:
            if self.retutn_x0_only:
                return X[0],y
            else:
                return X, y


