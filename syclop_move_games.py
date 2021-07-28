#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:15:51 2021

@author: orram
"""
import misc
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from tensorflow.keras.datasets import cifar10
import random
import SYCLOP_env as syc

def bad_res102(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    return dwnsmp

# load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
images, labels = trainX[:5], trainY[:5]
res = 8 
sample = 10 
mixed_state = True 
add_traject = True
trajectory_list=40
return_datasets=False 
add_seed = 20
show_fig = False
mix_res = False
bad_res_func = bad_res102
up_sample = False
img = images[4]
count = 0
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
        sensor = syc.Sensor(winx=52,winy=52,centralwinx=32,centralwiny=32,nchannels = 3,resolution_fun = lambda x: bad_res_func(x,(res,res)), resolution_fun_type = 'down')
    else:
        sensor = syc.Sensor(winx=32,winy=32,centralwinx=res//2,centralwiny=res//2,nchannels = 3,resolution_fun = lambda x: bad_res102(x,(res,res)), resolution_fun_type = 'down')
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])
    #Setting the coordinates to visit
    if type(trajectory_list) is int:
        if trajectory_list:
            np.random.seed(trajectory_list)
        starting_point = np.array([agent.max_q[0]//2,agent.max_q[1]//2])
        steps  = []
        for j in range(sample):
            steps.append(starting_point*1)
            starting_point += np.random.randint(-2,3,2) 

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
    #Create empty lists to store the syclops outputs
    imim=[]
    dimim=[]
    agent.set_manual_trajectory(manual_q_sequence=q_sequence)
    #Run Syclop for 20 time stepsבצדק זה היה ממש לא מתחשב!!

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
            