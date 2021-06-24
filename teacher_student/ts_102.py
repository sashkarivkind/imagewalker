#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Comparing student teacher methods -  using Knowledge Disillation (KD) and Feature Learning (FL)

Using Keras and Cifar 10 with syclop. 

A pervious comperision was made to distill knowledge with pytorch on cifar HR
images - there we found that the KD outperformed FL.
Another comparision was made with MNIST and pytorch with syclop on LR images - 
there we found that FL outperformed KD - the opposite!


"""
import sys
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
import numpy as np

import cv2
import misc
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from keras_utils import *
from misc import *

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy
#%%
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

trainX, testX = prep_pixels(trainX, testX)
#%%
#Define function for low resolution lens on syclop
def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh[:2], interpolation = cv2.INTER_CUBIC)
    return upsmp

def bad_res102(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_AREA)
    return dwnsmp

# import importlib
# importlib.reload(misc)
# from misc import Logger
import os 
import sys

######################## Network Parameters ##################################
st_parameters = {
'lr' : 5e-4,#float(sys.argv[1]),
'epochs' : 200,#int(sys.argv[2]),
"student_fst_learning" : 1,#int(sys.argv[4]), #The first learning stage of the student - number of epochs
'alpha': 0.9,#float(sys.argv[5]), #KD weights
'temp' : 20,#int(sys.argv[6]),   #KD weights
'beta' : 0.3 #float(sys.argv[7]), #features st weights
}

print('Run Parameters:', st_parameters)

class teacher_training(keras.Model):
    def __init__(self,teacher):
        super(teacher_training, self).__init__()
        self.teacher = teacher


    def compile(
        self,
        optimizer,
        metrics,
        loss_fn,

    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(teacher_training, self).compile(optimizer=optimizer, metrics=metrics)
        self.loss_fn = loss_fn


    def train_step(self, data):
        # Unpack data
        HR_data , y = data

        with tf.GradientTape() as tape:
            # Forward pass of student
            features, predictions = self.teacher(HR_data, training=True)

            # Compute losses
            loss = self.loss_fn(y, predictions)

        # Compute gradients
        trainable_vars = self.teacher.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}

        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        features, y_prediction = self.teacher(x, training=False)

        # Calculate the loss
        loss = self.loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}

        return results
    
    def call(self, data, training = False):
        x = data
        features, prediction = self.teacher(x, training = training)
        return features, prediction 


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        [syclop_data, HR_data] , y = data
        # Forward pass of teacher
        
        teacher_features, teacher_predictions = self.teacher.call(HR_data, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_features, student_predictions = self.student(syclop_data, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        student_features, y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
class feature_st(keras.Model):
    def __init__(self, student, teacher):
        super(feature_st, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        features_loss_fn,
        beta=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(feature_st, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.features_loss_fn = features_loss_fn
        self.beta = beta
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        [syclop_data, HR_data] , y = data
        # Forward pass of teacher
        teacher_features, teacher_predictions = self.teacher(HR_data, training=False)
        # layer_name = 'teacher_features'
        # intermediate_layer_model = keras.Model(inputs=model.input,
        #                                outputs=model.get_layer(layer_name).output)
        # intermediate_output = intermediate_layer_model(data)
        with tf.GradientTape() as tape:
            # Forward pass of student
            teacher_features, student_predictions = self.student(syclop_data, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            features_loss = self.features_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.beta * student_loss + (1 - self.beta) * features_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "features_loss": features_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        features, y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


def teacher(input_size = 32 ,dropout = 0.2):
    '''
    Takes only the first image from the burst and pass it trough a net that 
    aceives ~80% accuracy on full res cifar. 
    '''
    inputA = keras.layers.Input(shape=(input_size,input_size,3))

    
    # define CNN model
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(inputA)
    print(x1.shape)
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'valid')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)

    x1=keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'valid')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)

    x1=keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'valid')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)st_parameters['epochs']

    x1 = keras.layers.Flatten(name = 'teacher_features')(x1)
    print(x1.shape)
    x2 = keras.layers.Dense(10)(x1)
    print(x2.shape)
    model = keras.models.Model(inputs=inputA,outputs=[x1,x2], name = 'teacher')
    opt=tf.keras.optimizers.Adam(lr=1e-3)
    model = teacher_training(model)
    model.compile(
        optimizer=opt,
        loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def student(n_timesteps = 5, hidden_size = 256,input_size = 32, 
            concat = True, cnn_dropout = 0.4, rnn_dropout = 0.2):
    '''
    
    CNN RNN combination that extends the CNN to a network that achieves 
    ~80% accuracy on full res cifar.
    This student network outputs teacher = extended_cnn_one_img(n_timesteps = 1, input_size = st_parameters['epochs']32, dropout = 0.2)
    Parameters
    ----------
    n_timesteps : TYPE, optional
        DESCRIPTION. The default is 5.
    img_dim : TYPE, optional
        DESCRIPTION. The default is 32.
    hidden_size : TYPE, optional
        DESCRIPTION. The default is 128.teacher_network = teacher(input_size = 32, dropout = 0.2)
    input_size : TYPE, optional
        DESCRIPTION. The default is 32.epochs

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


    x1=keras.layers.TimeDistributed(keras.layers.Flatten(),name = 'cnn_student_features')(x1)
    print(x1.shape)
    if concat:
        x1 = keras.layers.Concatenate()([x1,inputB])
    else:
        x1 = x1
    print(x1.shape)

    # define LSTM model
    x1 = keras.layers.LSTM(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=rnn_dropout)(x1)
    print(x1.shape)
    x1 = keras.layers.Flatten()(x1)
    print(x1.shape)
    x1 = keras.layers.Dense(512, activation = 'relu', name = "student_features")(x1)
    print(x1.shape)
    
    x = keras.layers.Dense(10)(x1) #activation will be in the distiller
    model = keras.models.Model(inputs=[inputA,inputB],outputs=[x1,x], name = 'student{}'.format(concat))

    return model

res = 8
sample = 10
n_timesteps = sample

def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)
#%%

accur_dataset = pd.DataFrame()
#%%
teacher_network = teacher(input_size = 32, dropout = 0.2)

#%%
print('######################### TRAIN TEACHER ##############################')
teacher_history = teacher_network.fit(
                           trainX[:45000],
                           trainy[:45000],
                           batch_size = 64,
                           epochs = 15,#st_parameters['epochs'],
                           validation_data = (trainX[45000:],trainy[45000:]),
                           verbose = 0,
                           )
print('teacher test accuracy = ',teacher_history.history['val_sparse_categorical_accuracy'])
# accur_dataset['teacher_test'] = teacher_history.history['val_sparse_categorical_accuracy']
# accur_dataset['teacher_train'] = teacher_history.history['sparse_categorical_accuracy']
#%%
syclop_train_dataset, syclop_test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )
train_dataset_x, train_dataset_y = split_dataset_xy(syclop_train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(syclop_test_dataset)
#%%
print('######################### TRAIN STUDENT ##############################')

student_network = student(n_timesteps = sample, hidden_size = 128,input_size = 8, 
            concat = True, cnn_dropout = 0.4, rnn_dropout = 0.2)

#keras.utils.plot_model(student_network, expand_nested=True)
#%%
KD_student = keras.models.clone_model(student_network)
KD = Distiller(KD_student, teacher_network)

KD.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           distillation_loss_fn = keras.losses.KLDivergence(),
           alpha = st_parameters['alpha'],
           temperature = st_parameters['temp'])

KD_history = KD.fit([train_dataset_x,trainX[:45_000]],
        train_dataset_y,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (test_dataset_x, test_dataset_y), 
        verbose = 0,
        )

print('KD test accuracy = ',KD_history.history['val_sparse_categorical_accuracy'])
accur_dataset['KD_test'] = KD_history.history['val_sparse_categorical_accuracy']
accur_dataset['KD_train'] = KD_history.history['sparse_categorical_accuracy']
#%%
FL_student = keras.models.clone_model(student_network)
FL = feature_st(FL_student, teacher_network)

FL.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           features_loss_fn = keras.losses.MeanSquaredError(),
           beta = st_parameters['beta'],
           temperature = st_parameters['temp'])

FL_history = FL.fit([train_dataset_x,trainX[:45_000]],
        train_dataset_y,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (test_dataset_x, test_dataset_y), 
        verbose = 0,
        )

print('FL test accuracy = ',FL_history.history['val_sparse_categorical_accuracy'])
accur_dataset['FL_test'] = FL_history.history['val_sparse_categorical_accuracy']
accur_dataset['FL_train'] = FL_history.history['sparse_categorical_accuracy']
#%%
baseline_student = keras.models.clone_model(student_network)
baseline_model = teacher_training(baseline_student)
baseline_model.compile(
        optimizer=keras.optimizers.Adam(lr = st_parameters['lr']),
        loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
baseline_history = baseline_model.fit(train_dataset_x,
        train_dataset_y,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (test_dataset_x, test_dataset_y), 
        verbose = 0,
        )

print('Baseline test accuracy = ',baseline_history.history['val_sparse_categorical_accuracy'])
accur_dataset['baseline_test'] = baseline_history.history['val_sparse_categorical_accuracy']
accur_dataset['baseline_train'] = baseline_history.history['sparse_categorical_accuracy']
#%%
plt.figure()
#plt.plot(teacher_history.history['val_sparse_categorical_accuracy'], label = 'teacher')
plt.plot(KD_history.history['val_sparse_categorical_accuracy'], label = 'KD')
plt.plot(FL_history.history['val_sparse_categorical_accuracy'], label = 'FL')
plt.plot(baseline_history.history['val_sparse_categorical_accuracy'], label = 'baseline')
plt.legend()
plt.title('Comparing KD and FL teacher student models on cifar')
plt.savefig('KD_FL_cifar_syclop_{:.0e}_{}_{:.0e}.png'.format(st_parameters['alpha'],st_parameters['temp'], st_parameters['beta']))

#%%
print('################ Train Student - with pre-training ####################')
base_student = keras.models.clone_model(student_network)

base_model = teacher_training(base_student)
base_model.compile(
        optimizer=keras.optimizers.Adam(lr = st_parameters['lr']),
        loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
base_history = base_model.fit(train_dataset_x,
        train_dataset_y,
        batch_size = 64,
        epochs = st_parameters['student_fst_learning'],
        validation_data = (test_dataset_x,train_dataset_x),
        verbose = 0,
        )

print('Base test accuracy = ',base_history.history['val_sparse_categorical_accuracy'])

#%%
new_epochs = st_parameters['epochs'] - st_parameters['student_fst_learning']
if new_epochs < 0:
    new_epochs = 1
    print("ERROR - less than 1 epochs left after base learning!!!!!!!!")
print('############## Knowledge Distillation wt Pre-Trainiong ################')
KD_student_pt = keras.models.clone_model(base_student)
KD_pt = Distiller(KD_student_pt, teacher_network)

KD_pt.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           distillation_loss_fn = keras.losses.KLDivergence(),
           alpha = st_parameters['alpha'],
           temperature = st_parameters['temp'])

KD_pt_history = KD_pt.fit([train_dataset_x,trainX[:45_000]],
        train_dataset_y,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (test_dataset_x, test_dataset_y), 
        verbose = 0,
        )

print('KD pt test accuracy = ',KD_pt_history.history['val_sparse_categorical_accuracy'])
accur_dataset['KD_pt_test'] = base_history.history['val_sparse_categorical_accuracy'] + \
                                KD_pt_history.history['val_sparse_categorical_accuracy']
accur_dataset['KD_pt_train'] = base_history.history['sparse_categorical_accuracy'] + \
                                KD_pt_history.history['sparse_categorical_accuracy']
#%%
print('########################## Feature Learing ############################')
FL_pt_student = keras.models.clone_model(base_student)
FL_pt = feature_st(FL_pt_student, teacher_network)

FL_pt.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           features_loss_fn = keras.losses.MeanSquaredError(),
           beta = st_parameters['beta'],
           temperature = st_parameters['temp'])

FL_pt_history = FL_pt.fit([train_dataset_x,trainX[:45_000]],
        train_dataset_y,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (test_dataset_x, test_dataset_y), 
        verbose = 0,
        )


print('FL pt test accuracy = ',FL_pt_history.history['val_sparse_categorical_accuracy'])
accur_dataset['FL_pt_test'] = base_history.history['val_sparse_categorical_accuracy'] + \
                                FL_pt_history.history['val_sparse_categorical_accuracy']
accur_dataset['FL_pt_train'] = base_history.history['val_sparse_categorical_accuracy'] + \
                                FL_pt_history.history['sparse_categorical_accuracy']



accur_dataset.to_pickle('KD_FL_pt_cifar_syclop_{:.0e}_{}_{:.0e}.pkl'.format(st_parameters['alpha'],st_parameters['temp'], st_parameters['beta']))
#%%
plt.figure()
#plt.plot(teacher_history.history['val_sparse_categorical_accuracy'], label = 'teacher')
plt.plot(KD_history.history['val_sparse_categorical_accuracy'], label = 'KD')
plt.plot(FL_history.history['val_sparse_categorical_accuracy'], label = 'FL')
plt.plot(base_history.history['val_sparse_categorical_accuracy'], label = 'baseline')
plt.plot(accur_dataset['KD_pt_test'], label = 'KD_pt')
plt.plot(accur_dataset['FL_pt_test'], label = 'FL_pt')
plt.legend()
plt.title('Comparing KD and FL teacher student models on cifar')
plt.savefig('KD_FL_pt_cifar_syclop_{:.0e}_{}_{:.0e}.png'.format(st_parameters['alpha'],st_parameters['temp'], st_parameters['beta']))
