import os, re, time, json
# import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import sys

#sys.path.insert(1, '/home/labs/ahissarlab/arivkind/imagewalker')
#sys.path.insert(1, '/home/labs/ahissarlab/arivkind/imagewalker')
#sys.path.insert(1, '/home/bnapp/arivkindNet/imagewalker/')
# sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
# sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')

sys.path.insert(1, './..')
sys.path.insert(1, './../teacher_student')
import cv2
# try:
#     # %tensorflow_version only exists in Colab.
#     % tensorflow_version
#     2.
#     x
# except Exception:
#     pass
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10, cifar100

from matplotlib import pyplot as plt
# from keras_utils import create_cifar_dataset, split_dataset_xy
import argparse
# import tensorflow_datasets as tfds
import pdb
from split_keras_model import split_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from vanvalenlab_convolutional_recurrent import ConvGRU2D

from dataset_utils import Syclopic_dataset_generator, test_num_of_trajectories

print("Tensorflow version " + tf.__version__)

# (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# training_images = np.array([bad_res102(xx,(new_res,new_res)) for xx in training_images])
#
# validation_images = training_images[-5000:]
# validation_labels = training_labels[-5000:]
#
# training_images = training_images[:-5000]
# training_labels = training_labels[:-5000]


def preprocess_image_input(input_images):
    input_images = input_images.astype('float32')
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims


'''
Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
Input size is 224 x 224.
'''


def feature_extractor(inputs):
    feature_extractor = keras.layers.TimeDistributed(tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                              include_top=False,
                                                              weights=parameters['preload_weights']))
    print(parameters['trainable'])
    feature_extractor.trainable = parameters['trainable']
    A = feature_extractor(inputs)
    return A


'''
Defines final dense layers and subsequent softmax layer for classification.
'''


def classifier(inputs,n_classes=10,add_trajectory = False):
    print(add_trajectory)
    if add_trajectory:
        inputB = parameters['position_scale']*inputs[1]
        inputB = tf.expand_dims(inputB, 2)
        inputB = tf.expand_dims(inputB, 2)
        inputB = tf.tile(inputB, [1, 1, 7, 7, 1])
        print(inputB.shape)
        inputA = inputs[0]
        # print(inputs.shape)
    else:
        inputA = inputs[0]

    if parameters['conv_back_end']:

        if parameters['conv_rnn_type'] == 'lstm':
            Our_RNN_cell = keras.layers.ConvLSTM2D
        elif  parameters['conv_rnn_type'] == 'gru':
            Our_RNN_cell = ConvGRU2D
        else:
            error("not supported type of conv rnn cell")

        x = inputA
        if add_trajectory:
            x = keras.layers.Concatenate()([x,inputB])
        x = Our_RNN_cell(parameters['n_units1'], (3, 3), padding = 'same', return_sequences = True,name='convrnn1',dropout=0)(x)

        x = Our_RNN_cell(parameters['n_units2'], (3, 3), padding = 'same', return_sequences = True,name='convrnn2',dropout=0)(x)
        x = keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
        x = keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.AveragePooling1D(pool_size=(parameters['n_samples']))(x)
        print(x.shape)
    else:
        x = keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(inputA)
        x = keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        if add_trajectory:
            x = keras.layers.Concatenate()([x,inputB])
        x = tf.keras.layers.GRU(parameters['n_units1'],input_shape=(parameters['n_samples'], None),
                             return_sequences=True,
                             go_backwards=False)(x)
        x = tf.keras.layers.GRU(parameters['n_units2'],input_shape=(parameters['n_samples'], None),
                             return_sequences=True,
                             go_backwards=False)(x)
        x = tf.keras.layers.AveragePooling1D(pool_size=(parameters['n_samples']))(x)
        print(x.shape)


    x = tf.keras.layers.Dense(n_classes, activation="softmax", name="classification")(x)
    return x


'''
Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
Connect the feature extraction and "classifier" layers to build the model.
'''


def final_model(inputs,res=32,n_classes=10):

    inputB = inputs[1]
    inputs = inputs[0]
    resize = keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(224//res, 224//res)))(inputs)
    print(resize.shape)
    resnet_feature_extractor = feature_extractor(resize)
    print(parameters['add_trajectory'])
    classification_output = classifier((resnet_feature_extractor,inputB),n_classes=n_classes,add_trajectory=parameters['add_trajectory'])

    return classification_output


'''
Define the model and compile it. 
Use Stochastic Gradient Descent as the optimizer.
Use Sparse Categorical CrossEntropy as the loss function.
'''

def define_compile_split_model(res=32, metrics=['accuracy'], split_after_layer='pool1_pool',n_classes=10):
    resnet50 = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                              include_top=False,
                                                              weights=parameters['preload_weights'])
    resnet50_buttom,resnet50_top = split_model(resnet50, split_after_layer)
    del resnet50

    input_fe = tf.keras.layers.Input(shape=(res, res, 3))
    resize = tf.keras.layers.UpSampling2D(size=(224//res, 224//res))(input_fe)
    resnet_bottom_output = resnet50_buttom(resize)

    input_be = keras.layers.Input(shape=np.shape(resnet_bottom_output)[1:])
    resnet_top_output = resnet50_top(input_be)
    classification_output = classifier(resnet_top_output,n_classes=n_classes)

    fe_model = tf.keras.Model(inputs=input_fe, outputs=resnet_bottom_output)
    be_model = keras.models.Model(inputs=input_be, outputs=classification_output, name='backend')

    model = tf.keras.models.Sequential()
    model.add(fe_model)
    model.add(be_model)

    if parameters['optimizer'] == 'Adam':
        opt = tf.keras.optimizers.Adam(lr=parameters['lr'])
    elif parameters['optimizer'] == 'SGD':
        opt = tf.keras.optimizers.SGD(lr=parameters['lr'])
    else:
        error

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return model


def define_compile_model(res=32, metrics=['accuracy'], n_classes=10):
    inputs = tf.keras.layers.Input(shape=(parameters['n_samples'],res, res, 3))
    inputB = tf.keras.layers.Input(shape=(parameters['n_samples'],2))


    classification_output = final_model((inputs,inputB), res=res,n_classes=n_classes)

    model = tf.keras.Model(inputs=[inputs,inputB], outputs=classification_output)

    if parameters['optimizer'] == 'Adam':
        opt = tf.keras.optimizers.Adam(lr=parameters['lr'])
    elif parameters['optimizer'] == 'SGD':
        opt = tf.keras.optimizers.SGD(lr=parameters['lr'])
    else:
        error

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    # model.compile(optimizer='SGD',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=metrics)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--res', default=8, type=int, help='resolution')
    parser.add_argument('--n_classes', default=10, type=int, help='epochs')

    parser.add_argument('--epochs', default=100, type=int, help='num training epochs')
    parser.add_argument('--n_samples', default=5, type=int, help='num of samplese per trajectory')
    parser.add_argument('--trajectories_num', default=-1, type=int, help='number of trajectories to use')
    parser.add_argument('--broadcast', default=0, type=int,
                        help='1-integrate the coordinates by broadcasting them as extra dimentions, 2- add coordinates as an extra input')
    parser.add_argument('--early_stopper_option', default=0, type=int,
                        help='0- as for first submission 1-restore best weights, 2- restore best weights; stop according accuracy, not loss')

    parser.add_argument('--n_units1', default=1024, type=int, help='hidden layer1')
    parser.add_argument('--n_units2', default=512, type=int, help='hidden_layer2')

    parser.add_argument('--conv_rnn_type', default='gru', type=str, help='conv_rnn_type')
    parser.add_argument('--preload_weights', default='imagenet', type=str, help='weights to preload - default: imagenet, None for None')

    parser.add_argument('--style', default='brownian', type=str, help='choose syclops style of motion')
    parser.add_argument('--noise', default=0.15, type=float, help='added noise to the const_p_noise style')
    parser.add_argument('--max_length', default=5, type=int, help='choose syclops max trajectory length')
    parser.add_argument('--val_set_mult', default=5, type=int,
                        help='repetitions of validation dataset to reduce trajectory noise')

    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')


    parser.add_argument('--stopping_patience', default=10, type=int, help='stopping patience')
    parser.add_argument('--learning_patience', default=5, type=int, help='stopping patience')
    parser.add_argument('--conv_back_end', default=True)
    parser.add_argument('--trainable', default=True)
    parser.add_argument('--add_trajectory', default=0)
    parser.add_argument('--position_scale', default=1.0, type=float, help='added noise to the const_p_noise style')

    config = parser.parse_args()
    config = vars(config)
    print('config  ', config)

    parameters = config

    if parameters['preload_weights'] == 'None':
        parameters['preload_weights'] = None

    BATCH_SIZE = 32
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    res = parameters['res']

    # def bad_res102(img,res):
    #     sh=np.shape(img)
    #     dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    #     return dwnsmp

    # new_res = int(sys.argv[1])
    print('-----------setting resolution to {} ------'.format(res))

    model = define_compile_model(res=parameters['res'],n_classes=parameters['n_classes'],  metrics=['sparse_categorical_accuracy'])
    model.summary()

    # train_X = preprocess_image_input(training_images)
    # valid_X = preprocess_image_input(validation_images)
    if config['n_classes'] == 10:
        (trainX, trainY), (testX, testY)  = cifar10.load_data()
    elif config['n_classes'] == 100:
        (trainX, trainY), (testX, testY)  = cifar100.load_data()
    else:
        error

    images, labels = trainX, trainY

    # trainX=preprocess_image_input(trainX)
    # trainX=preprocess_image_input(testX)
    scale = 255
    preprocess_fun = lambda x: preprocess_image_input(scale*x)

    position_dim = (parameters['n_samples'], parameters['res'], parameters['res'], 2) if parameters['broadcast'] == 1 else (parameters['n_samples'], 2)
    movie_dim = (parameters['n_samples'], parameters['res'], parameters['res'], 3)

    def args_to_dict(**kwargs):
        return kwargs

    generator_params = args_to_dict(batch_size=BATCH_SIZE, movie_dim=movie_dim, position_dim=position_dim,
                                    n_classes=None, shuffle=True,
                                    prep_data_per_batch=True, one_hot_labels=False,
                                    res=parameters['res'],
                                    n_samples=parameters['n_samples'],
                                    mixed_state=True,
                                    n_trajectories=parameters['trajectories_num'],
                                    trajectory_list=0,
                                    broadcast=parameters['broadcast'],
                                    style=parameters['style'],
                                    max_length=parameters['max_length'],
                                    noise=parameters['noise'])
    print('preparing generators')

    # generator 2
    train_generator_classifier = Syclopic_dataset_generator(trainX[:-5000], labels[:-5000],  one_random_sample=False, preprocess_fun=preprocess_fun, **generator_params)
    # print(train_generator_classifier[0][0].shape)
    # print(train_generator_classifier[0][0][0].std(), train_generator_classifier[0][0][0].mean())
    # print(train_generator_classifier[0][0][1].std(), train_generator_classifier[0][0][1].mean())
    #
    # print(train_generator_classifier[0][0][0,0,:,:])
    # error
    val_generator_classifier = Syclopic_dataset_generator(trainX[-5000:].repeat(parameters['val_set_mult'], axis=0),
                                                          labels[-5000:].repeat(parameters['val_set_mult'], axis=0),  one_random_sample=False, preprocess_fun=preprocess_fun,
                                                          validation_mode=True, **generator_params)

    test_generator_classifier = Syclopic_dataset_generator(testX,
                                                          testY,  one_random_sample=False, preprocess_fun=preprocess_fun,
                                                          validation_mode=False, **generator_params)

    if parameters['early_stopper_option']==0:
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=config['learning_patience'],
                                       min_lr=0.5e-6,)
        early_stopper = EarlyStopping(min_delta=0.001, patience=config['stopping_patience'])
    elif parameters['early_stopper_option']==1:
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=config['learning_patience'],
                                       min_lr=0.5e-6,)
        early_stopper = EarlyStopping(min_delta=0.001, patience=config['stopping_patience'],restore_best_weights=True)
    elif parameters['early_stopper_option']==2:
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=config['learning_patience'],
                                       min_lr=0.5e-6, monitor='val_sparse_categorical_accuracy')
        early_stopper = EarlyStopping(min_delta=0.001, patience=config['stopping_patience'],monitor='val_sparse_categorical_accuracy',restore_best_weights=True)
    else:
        error

    history = model.fit(train_generator_classifier, epochs=parameters['epochs'], validation_data=val_generator_classifier,
                         verbose=2,callbacks=[lr_reducer, early_stopper])

    print('now testing:')
    for ii in range(5):
        evaluate_net = model.evaluate(test_generator_classifier,verbose=2)
    lsbjob = os.getenv('LSB_JOBID')
    lsbjob = '' if lsbjob is None else lsbjob
    run_name = 'res_rnn_baseline_' + str(parameters['n_classes']) + '_j' + lsbjob + '_t' + str(int(time.time()))
    
    model.save(run_name)
        # ev2= == label

# pdb.set_trace()