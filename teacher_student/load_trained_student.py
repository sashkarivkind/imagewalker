#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loading studant from saved model
"""
import os 
import sys
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
from feature_learning_utils import  student3,  write_to_file, full_learning_dataset_update,  net_weights_reinitializer
from keras_utils import create_cifar_dataset, split_dataset_xy

print(os.getcwd() + '/')
#%%

def load_student(path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/',  run_name = 'noname_j178_t1630240486'):


    temp_path = path + 'saved_models/{}_feature/'.format(run_name)
    home_folder = temp_path + '{}_saved_models/'.format(run_name)
    
    child_folder = home_folder + 'end_of_run_model/'
    
    
    #loading weights as numpy array
    numpy_weights_path = child_folder + '{}_numpy_weights/'.format(run_name)
    with open(numpy_weights_path + 'numpy_weights_{}'.format(run_name), 'rb') as file_pi:
        np_weights = pickle.load(file_pi)
        
    data = pd.read_pickle(path + 'feature_learning/summary_dataframe_feature_learning_full_train_103.pkl')
    parameters = data[data['this_run_name'] == run_name].to_dict('records')[0]
    
    
    numpy_student = student3(sample = int(parameters['max_length']), 
                       res = int(parameters['res']), 
                        activation = parameters['student_nl'],
                        dropout = parameters['dropout'], 
                        rnn_dropout = parameters['rnn_dropout'],
                        num_feature = int(parameters['num_feature']),
                       layer_norm = parameters['layer_norm_student'],
                       conv_rnn_type = parameters['conv_rnn_type'],
                       block_size = int(parameters['student_block_size']),
                       add_coordinates = parameters['broadcast'],
                       time_pool = parameters['time_pool'])
    layer_index = 0
    for layer in numpy_student.layers:
        if layer.name[:-2] == 'convLSTM':
            layer_name = layer.name
            saved_weights = [np_weights[layer_index], np_weights[layer_index+ 1], np_weights[layer_index+ 2]]
            numpy_student.get_layer(layer_name).set_weights(saved_weights)
            layer_index += 3
            
    return numpy_student