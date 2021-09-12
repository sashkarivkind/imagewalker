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

def create_trajectory(starting_point, sample = 5, style = 'brownian', noise = 0.15):
    steps = []
    phi = np.random.randint(0.1,2*np.pi) #the angle in polar coordinates
    speed = 0.8#np.abs(0.5 + np.random.normal(0,0.5))         #Constant added to the radios
    r = 3
    name_list = ['const direction + noise','ZigZag','spiral', 'brownian','degenerate']
    speed_noise = speed * 0.2
    phi_noise = 0.05
    x, y = starting_point[1], starting_point[0]
    steps.append([y,x])
    phi_speed =  (1/8)*np.pi
    old_style = style
    for j in range(sample-1):
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
            r += speed/2 + np.random.normal(-0.5,speed_noise)
            phi_noise = 0.1
            phi_speed = np.random.normal((2/4)*np.pi,(1/8)*np.pi)
            factor = 1#np.random.choice([-1,1])
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
        elif style == 'degenerate':
            r += speed + np.random.normal(-0.5,speed_noise)
        elif style == 'old':
            
            starting_point += np.random.randint(-2,3,2) 
            r = 0
            phi = 0
        x, y = starting_point[1] + int(r * np.cos(phi)), starting_point[0]+int(r * np.sin(phi))
        steps.append([y,x])
        
            
    return steps



    
#The Dataset formation

        
class Syclopic_dataset_generator(keras.utils.Sequence):

    def __init__(images, labels, res, sample = 5, mixed_state = True, add_traject = True,
                   trajectory_list=0,return_datasets=False, n_trajectories = 20, show_fig = False,
                   bad_res_func = bad_res102, up_sample = False, broadcast = 0,
                   style = 'brownian', noise = 0.15, max_length = 20):
    '''
    Creates a torch dataloader object of syclop outputs 
    from a list of images and labels.
    
    Parameters
    ----------
    images : List object holding the images to proces
    labels : List object holding the labels
    res : resolution dawnsampling factor - to be used in cv.resize(orig_img, res)
    sample: the number of samples to have in syclop
    mixed_state : if False, use the same trajectory on every image.
    return_datasets: rerutns datasets rather than dataloaders
    Returns
    -------
    train_dataloader, test_dataloader - torch DataLoader class objects

    '''
    if sample > max_length:
        max_length = sample
        print('max_length ({}) must be >= sample ({}), changed max_length to be == sample'.format(max_length, sample))
    count = 0
    ts_images = []
    dvs_images = []
    q_seq = []
    seed_list = []
    count = 0
    if mixed_state:
        np.random.seed(42)
        new_seed = 42

    #initiating syclop instance for generating sequences of images
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
        if n_trajectories:
            new_seed = random.randint(0,n_trajectories)
            np.random.seed(new_seed)    
        orig_img = img*1
        if img_num == 42:
            print('Are we Random?? ', np.random.randint(1, 20))

        #Set the padded image
        img=misc.build_cifar_padded(1./256*img)
        img_size = img.shape


        #Setting the coordinates to visit
        if type(trajectory_list) is int:
            if trajectory_list:
                np.random.seed(trajectory_list)
            starting_point = np.array([agent.max_q[0]//2,agent.max_q[1]//2])
            steps = create_trajectory(starting_point= starting_point, 
                                      sample = sample,
                                      style = style,
                                       noise = noise)

            if mixed_state:
                seed_list.append(new_seed)
                q_sequence = np.array(steps).astype(int)
            else:
                if count == 0:
                    q_sequence = np.array(steps).astype(int)
        else:
            q_sequence = np.array(trajectory_list[img_num]).astype(int)

        if count == 0 :
            print(q_sequence.shape)
            
        #Create empty lists to store the syclops outputs
        imim=[]
        dimim=[]

        agent.set_manual_trajectory(manual_q_sequence=q_sequence)
        for t in range(len(q_sequence)):
            agent.manual_act()
            sensor.update(scene, agent)
            imim.append(sensor.frame_view)
            dimim.append(sensor.dvs_view)


        imim = np.array(imim)        
        dimim = np.array(dimim)
        #Add current proccessed image to lists
        ts_images.append(imim)
        dvs_images.append(dimim)
        if broadcast==1:
            broadcast_place = np.ones(shape = [sample,res,res,2])
            for i in range(sample):
                broadcast_place[i,:,:,0] *= q_sequence[i,0]
                broadcast_place[i,:,:,1] *= q_sequence[i,1]
            q_seq.append(broadcast_place/img_size[0])
        else:
            q_seq.append(q_sequence/img_size[0])
        count += 1
    print(q_sequence)
    #pre pad all images to max_length
    for idx, image in enumerate(ts_images):
        image_base = np.zeros(shape = [max_length, res, res, 3])
        if broadcast:
            seq_base = np.zeros(shape = [max_length, res, res, 2])
        else:
            seq_base = np.zeros([max_length, 2])
        image_base[-len(imim):] = image
        seq_base[-len(q_sequence):] = q_seq[idx]
        ts_images[idx] = image_base * 1
        q_seq[idx] = seq_base * 1

    
    class cifar_dataset():
        def __init__(self, data, labels, add_traject = False, transform = None):

            self.data = data
            self.labels = labels

            self.add_traject = add_traject
            self.transform = transform
        def __len__(self):
            if self.add_traject: 
                return len(self.data[0]) 
            else: return len(self.data[0])


        def __getitem__(self, idx):
            '''
            args idx (int) :  index

            returns: tuple(data, label)
            '''
            if self.add_traject:
                img_data = self.data[0][idx] 
                traject_data = self.data[1][idx]
                label = self.labels[idx]
                return img_data, traject_data, label
            else:
                data = self.data[idx]



            if self.transform:
                data = self.transform(data)
                return data, label
            else:
                return data, label

        def dataset(self):
            return self.data
        def labels(self):
            return self.labels
        
    train_dataset = cifar_dataset(ts_train, train_labels,add_traject = add_traject)
    test_dataset = cifar_dataset(ts_val, val_labels,add_traject = add_traject)

    if return_datasets:
        if mixed_state:
            return train_dataset, test_dataset, seed_list
        else:
            return train_dataset, test_dataset

def mnist_split_dataset_xy(dataset,n_timesteps=5):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1)[...,np.newaxis],np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def split_dataset_xy(dataset,sample,one_random_sample=False, return_x1_only=False):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    if one_random_sample: #returning  one random sample from each sequence (used in baseline tests)
        data_len = np.shape(dataset_x1)[0]
        indx     = list(range(data_len))
        pick_sample = np.random.randint(sample,size=data_len)
        if return_x1_only:
            return np.array(dataset_x1)[indx,pick_sample,...], np.array(dataset_y) #todo: understand why we need a new axis here!
            # return np.array(dataset_x1)[indx,pick_sample,...,np.newaxis], np.array(dataset_y)
        else:
            return (np.array(dataset_x1)[indx,pick_sample,...,np.newaxis],np.array(dataset_x2)[:,:sample,:]),np.array(dataset_y) #todo: understand why there is an extra axis here??
    else:
        if return_x1_only:
            return np.array(dataset_x1),np.array(dataset_y)
        else:
            return (np.array(dataset_x1)[...,np.newaxis],np.array(dataset_x2)[:,:sample,:]),np.array(dataset_y) #todo: understand why there is an extra axis here??


