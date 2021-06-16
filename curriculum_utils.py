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
import pickle
import pandas as pd
import random

from sklearn.metrics import confusion_matrix

import importlib
importlib.reload(misc)

from mnist import MNIST

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


def create_mnist_dataset(images, labels, res, sample=5, mixed_state=True, add_traject=True, q_0=None, alpha=0,
                         trajectory_list=None,random_trajectories=False, return_datasets=False, add_seed=70000, show_fig=False,
                         mix_res=False, bad_res_func=None, up_sample=False):
    #mix_res = False, bad_res_func = bad_res102, up_sample = False):
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
        # create subplot to hold examples from the dataset
        fig, ax = plt.subplots(2, 5)
        i = 0  # indexises for the subplot for image and for syclop vision
    for img_num, img in enumerate(images):

        if add_seed:
            np.random.seed(random.randint(0, add_seed))

        if mix_res:
            res = random.randint(6, 10)
            if img_num >= 55000:
                res = res_orig
        orig_img = np.reshape(img, [28, 28])
        # Set the padded image
        img = misc.build_mnist_padded(1. / 256 * np.reshape(img, [1, 28, 28]))
        if img_num == 42:
            print('Are we random?', np.random.randint(1, 20))
        if show_fig:
            if count < 5:
                ax[0, i].imshow(orig_img)
                plt.title(labels[count])
        # Set the sensor and the agent
        scene = syc.Scene(image_matrix=img)
        if up_sample:
            sensor = syc.Sensor(winx=56, winy=56, centralwinx=28, centralwiny=28,
                                resolution_fun=lambda x: bad_res_func(x, (res, res)), resolution_fun_type='down')
        else:
            sensor = syc.Sensor(winx=56, winy=56, centralwinx=res // 2, centralwiny=res // 2,
                                resolution_fun=lambda x: bad_res_func(x, (res, res)), resolution_fun_type='down')

        agent = syc.Agent(max_q=[scene.maxx - sensor.hp.winx, scene.maxy - sensor.hp.winy])
        # Setting the coordinates to visit
        if trajectory_list is None:
            if img_num==0 or random_trajectories:
                starting_point = np.array([agent.max_q[0] // 2, agent.max_q[1] // 2])
                steps = []
                for j in range(sample):
                    steps.append(starting_point * 1)
                    starting_point += np.random.randint(-5, 5, 2)

                if mixed_state:
                    q_sequence = np.array(steps).astype(int)
                else:
                    if count == 0:
                        q_sequence = np.array(steps).astype(int)
            if q_0 is not None:
                q_sequence = (q_0*(1-alpha)+q_sequence*alpha).astype(int)
        else:
            q_sequence = np.array(trajectory_list[img_num]).astype(int)

        # Setting the resolution function - starting with the regular resolution

        # Create empty lists to store the syclops outputs
        imim = []
        dimim = []
        agent.set_manual_trajectory(manual_q_sequence=q_sequence)
        # Run Syclop for 20 time steps
        for t in range(len(q_sequence)):
            agent.manual_act()
            sensor.update(scene, agent)
            ############################################################################
            #############CHANGED FROM sensor.central_frame_view TO sensor.frame_view####
            ############################################################################
            imim.append(sensor.frame_view)
            dimim.append(sensor.dvs_view)
        # Create a unified matrix from the list
        if show_fig:
            if count < 5:
                ax[1, i].imshow(imim[0])
                plt.title(labels[count])
                i += 1

        imim = np.array(imim)
        dimim = np.array(dimim)
        # Add current proccessed image to lists
        ts_images.append(imim)
        dvs_images.append(dimim)
        q_seq.append(q_sequence)# / 128)
        count += 1

    if add_traject:  # If we add the trjectories the train list will become a list of lists, the images and the
        # corrosponding trajectories, we will change the dataset structure as well. Note the the labels stay the same.
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
        def __init__(self, data, labels, add_traject=False, transform=None):

            self.data = data
            self.labels = labels

            self.add_traject = add_traject
            self.transform = transform

        def __len__(self):
            if self.add_traject:
                return len(self.data[0])
            else:
                return len(self.data[0])

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

    train_dataset = mnist_dataset(ts_train, train_labels, add_traject=True)
    test_dataset = mnist_dataset(ts_val, val_labels, add_traject=True)
    batch = 64

    if return_datasets:
        return train_dataset, test_dataset