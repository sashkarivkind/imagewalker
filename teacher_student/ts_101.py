#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:48:11 2021

@author: orram
"""

import tensorflow as tf
import tensorflow.keras as keras

HR_img = keras.Input(shape = (28,28,1), name = 'HR_img')
model = keras.Sequential([keras.layers.Conv2D(16,3,activation = 'relu'),
                           keras.layers.MaxPool2D(),
                           keras.layers.Dropout(0.2),
                           keras.layers.Conv2D(32,3,activation = 'relu'),
                           keras.layers.MaxPool2D(),
                           keras.layers.Dropout(0.2),
                           keras.layers.Conv2D(16,3, activation = 'relu', name = 'teacher_features'), 
                           keras.layers.Flatten(),
                           keras.layers.Dense(10, activation = 'softmax'),
                           ],
                           name = 'teacher'
                           )
tmp_model = keras.Model(model.layers[0].input, model.layers[3].output)
