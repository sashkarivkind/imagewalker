#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:42:01 2021

@author: orram
"""
import tensorflow.keras as keras
import tensorflow as tf

def cnn_model(n_timesteps = 5):
    inputA = keras.layers.Input(shape=(n_timesteps,32,32,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    print(inputA[0].shape)
    x1 = inputA[0]
    print(x1.shape)
    # define CNN model
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(inputA[0,:,:,:])
    print(x1.shape)
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(0.2)(x1)

    x1=keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(0.2)(x1)

    x1=keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(0.2)(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1 = keras.layers.Flatten()(x1)
    x1 = keras.layers.Dense(10,activation="softmax")(x1)
    print(x1.shape)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x1)
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

res_list = [32,25,20,16,12,10,8]
test_accuracy = []
#for res in res_list:
res = 20
net = cnn_model()

class conv_net(tf.keras.Model):
  def __init__(self):
    super(conv_net, self).__init__(name='')
    

    self.conv1 = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')
    self.conv2 = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')
    self.max1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.drop1 = keras.layers.Dropout(0.2)

    self.conv3 = keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')
    self.conv4 = keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')
    self.max2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.drop2 = keras.layers.Dropout(0.2)
    
    self.conv5 = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')
    self.conv6 = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')
    self.max3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.drop3 = keras.layers.Dropout(0.2)
    
    self.flatten = keras.layers.Flatten()
    self.dense = keras.layers.Dense(10,activation="softmax")

  def call(self, input_tensor, training=False):
      inputA = input_tensor[0]
      inputB = input_tensor[1]
      print(inputA.shape)
      x = inputA[0]
      print(x.shape)
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.max1(x)
      x = self.drop1(x)
      
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.max2(x)
      x = self.drop2(x)
      
      x = self.conv5(x)
      x = self.conv6(x)
      x = self.max3(x)
      x = self.drop3(x)
      
      x = self.flatten(x)
      x = self.dense(x)
      
      return x
