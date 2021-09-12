import os, re, time, json
# import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import sys

sys.path.insert(1, '/home/labs/ahissarlab/arivkind/imagewalker')
sys.path.insert(1, '/home/bnapp/arivkindNet/imagewalker/')
# sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
# sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')


import cv2
# try:
#     # %tensorflow_version only exists in Colab.
#     % tensorflow_version
#     2.
#     x
# except Exception:
#     pass
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10

from matplotlib import pyplot as plt
from keras_utils import create_cifar_dataset, split_dataset_xy
import argparse
# import tensorflow_datasets as tfds
import pdb
print("Tensorflow version " + tf.__version__)


parser = argparse.ArgumentParser()


parser.add_argument('--res', default=32, type=int, help='resolution')
parser.add_argument('--epochs', default=10, type=int, help='num training epochs')
parser.add_argument('--sample', default=1, type=int, help='num of samplese per trajectory')
parser.add_argument('--trajectories_num', default=10, type=int, help='number of trajectories to use')
parser.add_argument('--broadcast', default=0, type=int, help='1-integrate the coordinates by broadcasting them as extra dimentions, 2- add coordinates as an extra input')
parser.add_argument('--style', default='brownain', type=str, help='choose syclops style of motion')
parser.add_argument('--noise', default=0.15, type=float, help='added noise to the const_p_noise style')
parser.add_argument('--max_length', default=1, type=int, help='choose syclops max trajectory length')

config = parser.parse_args()
config = vars(config)
print('config  ',config)

parameters = config

BATCH_SIZE = 32
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
res = parameters['res']

# def bad_res102(img,res):
#     sh=np.shape(img)
#     dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
#     return dwnsmp

# new_res = int(sys.argv[1])
print('-----------setting resolution to {} ------'.format( res))
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

if True:
    def feature_extractor(inputs):
        feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                                  include_top=False,
                                                                  weights='imagenet')(inputs)
        return feature_extractor


    '''
    Defines final dense layers and subsequent softmax layer for classification.
    '''


    def classifier(inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
        return x


    '''
    Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
    Connect the feature extraction and "classifier" layers to build the model.
    '''


    def final_model(inputs):
        resize = tf.keras.layers.UpSampling2D(size=(224//res, 224//res))(inputs)

        resnet_feature_extractor = feature_extractor(resize)
        classification_output = classifier(resnet_feature_extractor)

        return classification_output


    '''
    Define the model and compile it. 
    Use Stochastic Gradient Descent as the optimizer.
    Use Sparse Categorical CrossEntropy as the loss function.
    '''


    def define_compile_model():
        inputs = tf.keras.layers.Input(shape=(res, res, 3))

        classification_output = final_model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=classification_output)

        model.compile(optimizer='SGD',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model


    model = define_compile_model()
    model.summary()

# train_X = preprocess_image_input(training_images)
# valid_X = preprocess_image_input(validation_images)

(trainX, trainY), (testX, testY) = cifar10.load_data()
images, labels = trainX, trainY

for epoch in range(parameters['epochs']):
    train_dataset, test_dataset, seed_list = create_cifar_dataset(images, labels,res = res,
                                    sample = parameters['sample'], return_datasets=True,
                                    mixed_state = True,
                                    add_seed = parameters['trajectories_num'],
                                    trajectory_list = 0,
                                    broadcast=parameters['broadcast'],
                                    style = parameters['style'],
                                    max_length=parameters['max_length'],
                                    noise = parameters['noise'],
                                    )
    opt_dict = {'sample':parameters['sample'], 'one_random_sample':True, 'return_x1_only':True}

    train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset, **opt_dict)
    test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset, **opt_dict)
    # pdb.set_trace()
    train_dataset_x = preprocess_image_input(255*train_dataset_x)
    test_dataset_x = preprocess_image_input(255*test_dataset_x)
    history = model.fit(train_dataset_x, train_dataset_y, epochs=1, validation_data=(test_dataset_x, test_dataset_y),
                        batch_size=64, verbose=2)


loss, accuracy = model.evaluate(test_dataset_x, test_dataset_y, batch_size=64)

train_dataset, test_dataset, seed_list = create_cifar_dataset(images, labels, res=res,
                                                              sample=parameters['sample'], return_datasets=True,
                                                              mixed_state=True,
                                                              add_seed=parameters['trajectories_num'],
                                                              trajectory_list=0,
                                                              broadcast=parameters['broadcast'],
                                                              style=parameters['style'],
                                                              max_length=parameters['max_length'],
                                                              noise=parameters['noise'],
                                                              )

opt_dict = {'sample': parameters['sample'], 'one_random_sample': False, 'return_x1_only': True}

test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset, **opt_dict)
test_dataset_x = preprocess_image_input(255*test_dataset_x)

ev1=[]
for ii,(x,y) in enumerate(zip(test_dataset_x, test_dataset_y)):
    preds = model.predict(x)
    ev1.append( np.argmax(preds.sum(axis=0))  == y)
    if ii%100==0:
        print('step {}, intermediate comitee accuracy: {}'.format(ii,np.mean(ev1)))

    # ev2= == label

print('comitee accuracy: {}'.format(np.mean(ev1)))
# pdb.set_trace()