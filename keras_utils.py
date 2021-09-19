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
def create_mnist_dataset(images, labels, res, sample = 5, mixed_state = True, add_traject = True,
                   trajectory_list=None,return_datasets=False, add_seed = 20, show_fig = False,
                   mix_res = False, bad_res_func = bad_res102, up_sample = False):
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
    trajectory_list : uses a preset trajectory from the list.
    return_datasets: rerutns datasets rather than dataloaders
    add_seed : creates a random seed option to have a limited number of random
               trajectories, default = 20 (number of trajectories)
    show_fig : to show or not an example of the dataset, defoult = False
    mix_res  : Weather or not to create a mix of resolution in each call to
                the dataset, to use to learn if the network is able to learn
                mixed resolution to gain better performance in the lower res 
                part. default = 
    bed_res_func : The function that creats the bad resolution images 
    up_sample    : weather the bad_res_func used up sampling or not, it changes the central view 
                    values. 
    
    Returns
    -------
    train_dataloader, test_dataloader - torch DataLoader class objects

    '''
    count = 0
    ts_images = []
    dvs_images = []
    q_seq = []
    count = 0
    res_orig = res * 1 
    if show_fig:
        #create subplot to hold examples from the dataset
        fig, ax = plt.subplots(2,5)
        i = 0 #indexises for the subplot for image and for syclop vision
    for img_num,img in enumerate(images):
        
            
        if add_seed:
            np.random.seed(random.randint(0,add_seed))        
        
        if mix_res:
            res = random.randint(6,10)
            if img_num >= 55000:
                res = res_orig
        orig_img = np.reshape(img,[28,28])
        #Set the padded image
        img=misc.build_mnist_padded(1./256*np.reshape(img,[1,28,28]))
        if img_num == 42:
            print('Are we random?', np.random.randint(1,20))
        if show_fig:
            if count < 5:
                ax[0,i].imshow(orig_img) 
                plt.title(labels[count])
        #Set the sensor and the agent
        scene = syc.Scene(image_matrix=img)
        if up_sample:
            sensor = syc.Sensor(winx=56,winy=56,centralwinx=28,centralwiny=28,
                                resolution_fun = lambda x: bad_res_func(x,(res,res)), resolution_fun_type = 'down')
        else:
            sensor = syc.Sensor(winx=56,winy=56,centralwinx=res//2,centralwiny=res//2,
                                resolution_fun = lambda x: bad_res_func(x,(res,res)), resolution_fun_type = 'down')

        agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])
        #Setting the coordinates to visit
        if trajectory_list is None:
            starting_point = np.array([agent.max_q[0]//2,agent.max_q[1]//2])
            steps  = []
            for j in range(sample):
                steps.append(starting_point*1)
                starting_point += np.random.randint(-5,5,2) 

            if mixed_state:
                q_sequence = np.array(steps).astype(int)
            else:
                if count == 0:
                    q_sequence = np.array(steps).astype(int)
        else:
            q_sequence = np.array(trajectory_list[img_num]).astype(int)
        
        #Setting the resolution function - starting with the regular resolution
        
        #Create empty lists to store the syclops outputs
        imim=[]
        dimim=[]
        agent.set_manual_trajectory(manual_q_sequence=q_sequence)
        #Run Syclop for 20 time steps
        for t in range(len(q_sequence)):
            agent.manual_act()
            sensor.update(scene, agent)
            ############################################################################
            #############CHANGED FROM sensor.central_frame_view TO sensor.frame_view####
            ############################################################################
            imim.append(sensor.frame_view)
            dimim.append(sensor.dvs_view)
        #Create a unified matrix from the list
        if show_fig:
            if count < 5:
                ax[1,i].imshow(imim[0]) 
                plt.title(labels[count])
                i+=1
            

        imim = np.array(imim)
        dimim = np.array(dimim)
        #Add current proccessed image to lists
        ts_images.append(imim)
        dvs_images.append(dimim)
        q_seq.append(q_sequence/128)
        count += 1
        

    
    if add_traject: #If we add the trjectories the train list will become a list of lists, the images and the 
        #corrosponding trajectories, we will change the dataset structure as well. Note the the labels stay the same.
        ts_train = (ts_images[:55000], q_seq[:55000]) 
        train_labels = labels[:55000]
        ts_val = (ts_images[55000:], q_seq[55000:])
        val_labels = labels[55000:]

    else:
        ts_train = ts_images[:55000]
        train_labels = labels[:55000]
        ts_val = ts_images[55000:]
        val_labels = labels[55000:]

    dvs_train = dvs_images[:55000]
    dvs_val = dvs_images[55000:]
    
    class mnist_dataset():
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
        
    train_dataset = mnist_dataset(ts_train, train_labels,add_traject = True)
    test_dataset = mnist_dataset(ts_val, val_labels,add_traject = True)
    batch = 64

    if return_datasets:
        return train_dataset, test_dataset

    
#The Dataset formation
def print_traject(images, labels, res, sample = 5, mixed_state = True, add_traject = True,
                   trajectory_list=0,return_datasets=False, add_seed = True, show_fig = False,
                   bad_res_func = bad_res102, up_sample = False):
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
    count = 0
    ts_images = []
    dvs_images = []
    q_seq = []
    count = 0
    if show_fig:
        #create subplot to hold examples from the dataset
        fig, ax = plt.subplots(2,5)
        i = 0 #indexises for the subplot for image and for syclop vision
    for img_num,img in enumerate(images):
        if add_seed:
            np.random.seed(random.randint(0,add_seed))    
        orig_img = img*1
        #Set the padded image
        img=misc.build_cifar_padded(1./256*img)
        img_size = img.shape
        if img_num == 42:
            print('Are we Random?? ', np.random.randint(1,20))
        if show_fig:
            if count < 5:
                ax[0,i].imshow(orig_img) 
                plt.title(labels[count])
        #Set the sensor and the agent
        scene = syc.Scene(image_matrix=img)
        if up_sample:
            sensor = syc.Sensor(winx=56,winy=56,centralwinx=32,centralwiny=32,nchannels = 3,resolution_fun = lambda x: bad_res_func(x,(res,res)), resolution_fun_type = 'down')
        else:
            sensor = syc.Sensor(winx=56,winy=56,centralwinx=res//2,centralwiny=res//2,nchannels = 3,resolution_fun = lambda x: bad_res102(x,(res,res)), resolution_fun_type = 'down')
        agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])
        #Setting the coordinates to visit
        if type(trajectory_list) is int:
            if trajectory_list:
                np.random.seed(trajectory_list)
            starting_point = np.array([agent.max_q[0]//2,agent.max_q[1]//2])
            steps  = []
            for j in range(sample):
                steps.append(starting_point*1)
                starting_point += np.random.randint(-5,5,2) 

            if mixed_state:
                q_sequence = np.array(steps).astype(int)
            else:
                if count == 0:
                    q_sequence = np.array(steps).astype(int)
        else:
            q_sequence = np.array(trajectory_list[img_num]).astype(int)

        if count == 0 :
            print(q_sequence.shape)
        print(q_sequence)
        
        
def create_cifar_dataset(images, labels, res, sample = 5, mixed_state = True, add_traject = True,
                   trajectory_list=0,return_datasets=False, add_seed = True, show_fig = False,
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
    if show_fig:
        #create subplot to hold examples from the dataset
        fig, ax = plt.subplots(2,5)
        i = 0 #indexises for the subplot for image and for syclop vision
    
    for img_num,img in enumerate(images):
        if add_seed:
            new_seed = random.randint(0,add_seed)
            np.random.seed(new_seed)    
        orig_img = img*1
        #Set the padded image
        img=misc.build_cifar_padded(1./256*img)
        img_size = img.shape
        if img_num == 42:
            print('Are we Random?? ', np.random.randint(1,20))
        if show_fig:
            if count < 5:
                ax[0,i].imshow(orig_img) 
                plt.title(labels[count])
        #Set the sensor and the agent
        scene = syc.Scene(image_matrix=img)
        if up_sample:
            sensor = syc.Sensor(winx=56,winy=56,centralwinx=32,centralwiny=32,nchannels = 3,resolution_fun = lambda x: bad_res_func(x,(res,res)), resolution_fun_type = 'down')
        else:
            sensor = syc.Sensor(winx=32,winy=32,centralwinx=res//2,centralwiny=res//2,nchannels = 3,resolution_fun = lambda x: bad_res102(x,(res,res)), resolution_fun_type = 'down')
        agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])
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
        #Run Syclop for 20 time steps

        for t in range(len(q_sequence)):
            agent.manual_act()
            sensor.update(scene, agent)
            ############################################################################
            #############CHANGED FROM sensor.central_frame_view TO sensor.frame_view####
            ############################################################################
            imim.append(sensor.frame_view)
            dimim.append(sensor.dvs_view)
        #Create a unified matrix from the list
        if show_fig:
            if count < 5:
                ax[1,i].imshow(imim[0]) 
                plt.title(labels[count])
                i+=1
            
        #imim shape [sample,res,res,3]
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
        if broadcast==1:
            seq_base = np.zeros(shape = [max_length, res, res, 2])
        else:
            seq_base = np.zeros([max_length, 2])
        image_base[-len(imim):] = image
        seq_base[-len(q_sequence):] = q_seq[idx]
        ts_images[idx] = image_base * 1
        q_seq[idx] = seq_base * 1
        
    if add_traject: #If we add the trjectories the train list will become a list of lists, the images and the 
        #corrosponding trajectories, we will change the dataset structure as well. Note the the labels stay the same.
        ts_train = (ts_images[:45000], q_seq[:45000]) 
        train_labels = labels[:45000]
        ts_val = (ts_images[45000:], q_seq[45000:])
        val_labels = labels[45000:]

    else:
        ts_train = ts_images[:45000]
        train_labels = labels[:45000]
        ts_val = ts_images[45000:]
        val_labels = labels[45000:]

    dvs_train = dvs_images[:45000]
    dvs_val = dvs_images[45000:]
    
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


def write_to_file(history, net,paramaters, dataset_number):
    file_name = 'summary_file_{}_{}.txt'.format(net.name, dataset_number)
    if os.path.isfile('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}'.format(file_name)):
        file = open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}'.format(file_name), 'a')
    else:
        file = open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}'.format(file_name), 'x')
        
    
    from datetime import datetime
    file.write('#####################\n')
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")
    file.write(now + '\n')
    file.write(str(paramaters) + "\n")
    max_test = max(history.history['val_sparse_categorical_accuracy'])
    max_location = np.argmax(history.history['val_sparse_categorical_accuracy'])
    mean_last_20 = np.mean(history.history['val_sparse_categorical_accuracy'][-20:])
    max_train = max((history.history['sparse_categorical_accuracy']))
    when_train_reached_test_max = np.where(np.array(history.history['sparse_categorical_accuracy']) > max_test )
    if len(when_train_reached_test_max)>1:
        when_train_reached_test_max = when_train_reached_test_max[0]
    summary_of_run = "max_test = {}, max_location = {}, mean_last_20 = {}, max_train = {}, when_train_reached_test_max = {}\n".\
                        format(max_test,max_location,mean_last_20,max_train,when_train_reached_test_max)
        
    file.write(summary_of_run)
    file.close()

def dataset_update(history, net, parameters, dataset_number):
    file_name = 'summary_dataframe_{}_{}.pkl'.format(net.name, dataset_number)
    if os.path.isfile('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}'.format(file_name)):
        dataframe = pd.read_pickle('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}'.format(file_name))
    else:
        dataframe = pd.DataFrame()
    
    values_to_add = parameters
    values_to_add['max_test'] = max(history.history['val_sparse_categorical_accuracy'])
    values_to_add['max_location'] = np.argmax(history.history['val_sparse_categorical_accuracy'])
    values_to_add['mean_last_20'] = np.mean(history.history['val_sparse_categorical_accuracy'][-20:])
    values_to_add['max_train'] = max((history.history['sparse_categorical_accuracy']))
    values_to_add['when_train_reached_test_max'] = np.where(np.array(history.history['sparse_categorical_accuracy']) > parameters["max_test"] )
    if len(values_to_add['when_train_reached_test_max'])>1:
        values_to_add['when_train_reached_test_max'] = values_to_add['when_train_reached_test_max'][0]
        
    
    dataframe = dataframe.append(values_to_add, ignore_index = True)

    dataframe.to_pickle('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}'.format(file_name))
    
#######################################################################################################################
############################### KERAS NETWORKS ########################################################################
###############################                #####################################(base) orram@orram-Latitude-3400:~$ scp /home/orram/Docum###################################
#######################################################################################################################
def rnn_model(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 3, concat = True):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
    print(x1.shape)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)
    # define LSTM model
    x = keras.layers.GRU(cell_size,input_shape=(n_timesteps, None),return_sequences=False)(x)
    print(x.shape)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'rnn_model_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def extended_rnn_model(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True):
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
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x1)
    print(x1.shape)


    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)

    # define LSTM model
    x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),return_sequences=True)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'extended_rnn_model_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


    
def rnn_model_dense(n_timesteps = 5, gru_size = 128):
    inputA = keras.layers.Input(shape=(n_timesteps,28,28,1))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)

    x = keras.layers.Concatenate()([x1,inputB])
    print(x.shape)
    x=keras.layers.TimeDistributed(keras.layers.Dense(x.shape[2]))(x)
    print(x.shape)
    # define LSTM model
    x = keras.layers.GRU(gru_size,input_shape=(n_timesteps, None),return_sequences=False)(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'rnn_model_dense')
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def rnn_model_concat_conv(n_timesteps = 5, gru_size = 128):
    #NEED TO DO
    inputA = keras.layers.Input(shape=(n_timesteps,28,28,1))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
    x2=keras.layers.TimeDistributed(keras.layers.Conv2D(5,(3,3),activation='relu'))(np.ones([3,3,5]))
    print(x2.shape)
    print(x1.shape)
    print(inputB[:,:,0].shape)
    print(inputB[:,:,0]*np.ones([3,3,5]).shape)
    x1 = keras.layers.Concatenate(axis = 4)([x1,inputB*np.ones([3,3,2])])
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)

    x = x1 
    print(x.shape)
    x=keras.layers.TimeDistributed(keras.layers.Dense(x.shape[2]))(x)
    print(x.shape)
    # define LSTM model
    x = keras.layers.GRU(gru_size,input_shape=(n_timesteps, None),return_sequences=False)(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'rnn_model_concat_conv')
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def rnn_model_concat_same_length(n_timesteps = 5, gru_size = 128):
   # NEED TO DO!
    
    
    inputA = keras.layers.Input(shape=(n_timesteps,28,28,1))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    x2 = keras.layers.Concatenate()([inputB,inputB])
    for i in range(x1.shape[-1]//2 - x2.shape[-1]):
        x2 = keras.layers.Concatenate()([x2,inputB])
    
    print(x2.shape)
    x = keras.layers.Concatenate()([x1,x2])
    print(x.shape)
    x=keras.layers.TimeDistributed(keras.layers.Dense(x.shape[2]))(x)
    print(x.shape)
    # define LSTM model
    x = keras.layers.GRU(gru_size,input_shape=(n_timesteps, None),return_sequences=False)(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'rnn_model_concat_same_length')
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    return model
    
    
def simple_vanila_model(n_timesteps = 5, cell_size = 128):
    inputA = keras.layers.Input(shape=(n_timesteps,28,28,1))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)

    x = x1
    print(x.shape)
    # define LSTM model
    x = keras.layers.SimpleRNN(cell_size,input_shape=(n_timesteps, None),return_sequences=False)(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'simple_vanila_model')
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
    

def cnn_one_img(n_timesteps = 5, input_size = 28, input_dim = 1):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    print(inputA[:,0,:,:,:].shape)
    # define CNN model
    x1=keras.layers.Conv2D(16,(3,3),activation='relu')(inputA[:,0,:,:,:])
    x1=keras.layers.BatchNormalization()(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Conv2D(32,(3,3),activation='relu')(x1)
    x1=keras.layers.BatchNormalization()(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)

    x1=keras.layers.Conv2D(16,(3,3),activation='relu')(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1=keras.layers.Flatten()(x1)
    print(x1.shape)

    x = keras.layers.Dense(10,activation="softmax")(x1)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'cnn_one_img')
    opt=tf.keras.optimizers.Adam(lr=1e-4)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def extended_cnn_one_img(n_timesteps = 5, input_size = 32 ,dropout = 0.2):
    '''
    Takes only the first image from the burst and pass it trough a net that 
    aceives ~80% accuracy on full res cifar. 
    '''
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    print(inputA[:,0,:,:,:].shape)
    # define CNN model
    index = np.random.randint(0,n_timesteps)
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(inputA[:,index,:,:,:])
    print(x1.shape)
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)

    x1=keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)

    x1=keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1 = keras.layers.Flatten()(x1)
    x1 = keras.layers.Dense(10,activation="softmax")(x1)
    print(x1.shape)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x1, name = 'extended_cnn_one_img')
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


            
def low_features_rnn_model(n_timesteps = 5, cell_size = 128, input_size = 32, concat = True, low_cat = False):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    
    h_coor = 0
    hold_slices = []
    gru1 = keras.layers.GRU(4*4*16,input_shape=(n_timesteps, None),return_sequences=True)
    for i in range(8):
        v_coor =0
        v_slices = []
        for j in range(8):
            slice_ = inputA[:,:,v_coor:v_coor+4,h_coor:h_coor+4,:]
            slice_ = keras.layers.TimeDistributed(keras.layers.Flatten())(slice_)
            if low_cat:
                slice_ = keras.layers.Concatenate()([slice_, inputB])
            slice_ = gru1(slice_)
            #print(slice_.shape)
            slice_ = keras.layers.Reshape((n_timesteps, 4, 4, 16))(slice_)
            #print(slice_.shape)
            v_slices.append(slice_)
            v_coor += 4
        h_coor += 4
        hold_slices.append(v_slices)
    
    h = []
    for v_slice in hold_slices:
        v = keras.layers.Concatenate(axis = 2)(v_slice)
        #print(v.shape)
        h.append(v)
    h = keras.layers.Concatenate(axis = 3)(h)
    print(h.shape)
            
    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(h)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization(inputA))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)
 
    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)
    # define LSTM model
    x = keras.layers.GRU(cell_size,input_shape=(n_timesteps, None),return_sequences=False)(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'low_features_rnn_model_{}_{}'.format(concat,low_cat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def extended_low_features_rnn_model(n_timesteps = 5, cell_size = 128, input_size = 32, concat = True, low_cat = False):
    inputA = keras.layers.Input(shapinputAe=(n_timesteps,input_size,input_size,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    
    h_coor = 0
    hold_slices = []
    gru1 = keras.layers.GRU(4*4*16,input_shape=(n_timesteps, None),return_sequences=True)
    for i in range(8):
        v_coor =0
        v_slices = []
        for j in range(8):
            slice_ = inputA[:,:,v_coor:v_coor+4,h_coor:h_coor+4,:]
            slice_ = keras.layers.TimeDistributed(keras.layers.Flatten())(slice_)
            if low_cat:
                slice_ = keras.layers.Concatenate()([slice_, inputB])
            slice_ = gru1(slice_)
            #print(slice_.shape)
            slice_ = keras.layers.Reshape((n_timesteps, 4, 4, 16))(slice_)
            #print(slice_.shape)
            v_slices.append(slice_)
            v_coor += 4
        h_coor += 4
        hold_slices.append(v_slices)
    
    h = []
    for v_slice in hold_slices:
        v = keras.layers.Concatenate(axis = 2)(v_slice)
        #print(v.shape)
        h.append(v)
    h = keras.layers.Concatenate(axis = 3)(h)
    print(h.shape)
            
    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same'))(h)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)
 
    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)
    # define LSTM model
    x = keras.layers.GRU(cell_size,input_shape=(n_timesteps, None),return_sequences=False)(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'extended_low_features_rnn_model_{}_{}'.format(concat,low_cat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def no_cnn(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 1, concat = True):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(inputA)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)
    # define LSTM model
    x = keras.layers.GRU(cell_size,input_shape=(n_timesteps, None),return_sequences=False, recurrent_dropout=0.1)(x)
    print(x.shape)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'no_cnn_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def no_cnn_dense(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 1, concat = True):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(extended_cnn_one_imgshape=(n_timesteps,2))
    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(inputA)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)
    x = keras.layers.TimeDistributed(keras.layers.Dense(cell_size,activation="relu"))(x)
    # define LSTM model
    x = keras.layers.GRU(cell_size,input_shape=(n_timesteps, None),return_sequences=False, recurrent_dropout=0.1)(x)
    
    print(x.shape)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'no_cnn_dense_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def no_cnn_low_features_model(n_timesteps = 5, cell_size = 128, input_size = 32, input_dim = 3,concat = True, low_cat = False):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    
    h_coor = 0
    hold_slices = []
    gru1 = keras.layers.GRU(4*4*16,input_shape=(n_timesteps, None),return_sequences=True)
    print(inputA.shape)
    for i in range(4):
        v_coor =0
        v_slices = []
        for j in range(4):
            slice_ = inputA[:,:,v_coor:v_coor+1,h_coor:h_coor+1,:]
            print(v_coor+1)
            slice_ = keras.layers.TimeDistributed(keras.layers.Flatten())(slice_)
            if low_cat:
                slice_ = keras.layers.Concatenate()([slice_, inputB])
            slice_ = gru1(slice_)
            #print(slice_.shape)
            slice_ = keras.layers.Reshape((n_timesteps, 1, 1, 4*4*16))(slice_)
            #print(slice_.shape)cnn_dropout
            v_slices.append(slice_)
            v_coor += 1
        h_coor += 1
        hold_slices.append(v_slices)
    
    h = []
    for v_slice in hold_slices:
        v = keras.layers.Concatenate(axis = 2)(v_slice)
        #print(v.shape)
        h.append(v)
    h = keras.layers.Concatenate(axis = 3)(h)
    print('h shape:', h.shape)
            
    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(h)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print('x',x.shape)
    x = keras.layers.TimeDistributed(keras.layers.Dense(cell_size,activation="relu"))(x)
    # define GRU model
    x = keras.layers.GRU(cell_size,input_shape=(n_timesteps, None),return_sequences=False, recurrent_dropout=0.2)(x)
    
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'low_features_rnn_model_{}_{}'.format(concat,low_cat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def low_features_middle_cnn_model(n_timesteps = 5, cell_size = 128, input_size = 32, input_dim = 3,concat = True, low_cat = False):
    '''
    concat = True, low_cat = False GRU cell = 4*4*16 dropout_cnn = 0.2, dropout_rnn = 0.2 RESULTS = 57 train 54 val
    
    
    '''
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    
    h_coor = 0
    hold_slices = []
    gru1 = keras.layers.GRU(4*4*16,input_shape=(n_timesteps, None),return_sequences=True)
    print(inputA.shape)
    for i in range(4):
        v_coor =0
        v_slices = []
        for j in range(4):
            slice_ = inputA[:,:,v_coor:v_coor+1,h_coor:h_coor+1,:]
            slice_ = keras.layers.TimeDistributed(keras.layers.Flatten())(slice_)
            if low_cat:
                slice_ = keras.layers.Concatenate()([slice_, inputB])
            slice_ = gru1(slice_)
            #print(slice_.shape)
            slice_ = keras.layers.Reshape((n_timesteps, 1, 1, 4*4*16))(slice_)
            #print(slice_.shape)
            v_slices.append(slice_)
            v_coor += 1
        h_coor += 1
        hold_slices.append(v_slices)
    
    h = []
    for v_slice in hold_slices:
        v = keras.layers.Concatenate(axis = 2)(v_slice)
        #print(v.shape)
        h.append(v)
    h = keras.layers.Concatenate(axis = 3)(h)
    print('h shape:', h.shape)
    
    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(h)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    print(x1.shape)
    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print('x',x.shape)
    x = keras.layers.TimeDistributed(keras.layers.Dense(cell_size,activation="relu"))(x)
    # define GRU model
    x = keras.layers.GRU(cell_size,input_shape=(n_timesteps, None),return_sequences=False, recurrent_dropout=0.2)(x)
    
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'low_features_rnn_model_{}_{}'.format(concat,low_cat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def convgru(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 1, concat = False):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define LSTM model
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(inputA)
    print(x.shape)
    # x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(x)
    # print(x.shape)
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(x)
    print(x.shape)
    x = keras.layers.Flatten()(x)
    print(x.shape)
    if concat:
        x = keras.layers.Concatenate()([x,inputB])
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'ConvLSTM_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def cnn_lstm(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True, cnn_dropout = 0.4, rnn_dropout = 0.2):
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
    x = keras.layers.LSTM(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=rnn_dropout)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'cnn_lstm_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=5e-4)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
############################################################################################################3
############################# TEACHER STUDENT NETWORKS #####################################################
############################################################################################################
def HRcnn(input_size = 28, input_dim = 1):
    inputA = keras.layers.Input(shape=(input_size, input_size, input_dim))
    
    x1 = keras.layers.Conv2D(16, kernel_size = 3, activation = 'relu')(inputA)
    #x = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.MaxPool2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu')(x1)
    #x = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.MaxPool2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Conv2D(16, kernel_size = 3, activation = 'relu', name = 'teacher_features')(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    
    x = keras.layers.Flatten()(x1)
    
    x = keras.layers.Dense(10, activation = 'softmax')(x)
    
    model = keras.models.Model(inputs = inputA, outputs = x, name = 'HR_CNN')
    
    opt = tf.keras.optimizers.Adam(lr = 3e-3)
    
    model.compile( optimizer = opt,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['sparse_categorical_accuracy'])
    
    return model

# def student_loss(label, teacher_features, student_outpus):
    
#     student_features, student_pred = student_outputs
    
#     feature_loss = keras
    
    
def StudentCNN(input_size = 28, input_dim = 1):
    inputA = keras.layers.Input(shape=(input_size, input_size, input_dim))
    
    x1 = keras.layers.Conv2D(16, kernel_size = 3, activation = 'relu')(inputA)
    #x = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.MaxPool2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu')(x1)
    #x = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.MaxPool2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Conv2D(16, kernel_size = 3, activation = 'relu', name = 'student_feature')(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    
    x = keras.layers.Flatten()(x1)
    
    x = keras.layers.Dense(10, activation = 'softmax')(x)
    
    model = keras.models.Model(inputs = inputA, outputs = [x1,x], name = 'StudentCNN')
    
    opt = tf.keras.optimizers.Adam(lr = 3e-3)
    
    model.compile( optimizer = opt,
                  loss = ['mean_squared_error','sparse_categorical_crossentropy'],
                  metrics = ['mean_squared_error','sparse_categorical_accuracy'])
    
    return model

    