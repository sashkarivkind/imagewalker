#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 20:44:36 2021

@author: orram
"""
import pickle 
import numpy as np
import os 
path = '/home/labs/ahissarlab/orra/imagewalker/teacher_student/'
feature_data_path = path +'feature_data/'
if os.path.exists(feature_data_path + 'train_features_{}'.format('cnn3')):
    print('found data')
    train_data = np.array(pickle.load(open(feature_data_path + 'train_features_{}'.format('cnn3'),'rb')))
 