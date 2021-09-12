# %% [markdown] {"id":"FStp_vbUkRz5"}
#
#
#
# # Transfer Learning
# In this notebook, we will perform transfer learning to train CIFAR-10 dataset on ResNet50 model available in Keras.
#
#

# %% [markdown] {"id":"qpiJj8ym0v0-"}
# ## Imports

# %% [code] {"id":"AoilhmYe1b5t","execution":{"iopub.status.busy":"2021-08-02T17:27:37.867617Z","iopub.execute_input":"2021-08-02T17:27:37.868472Z","iopub.status.idle":"2021-08-02T17:27:45.661016Z","shell.execute_reply.started":"2021-08-02T17:27:37.868405Z","shell.execute_reply":"2021-08-02T17:27:45.659904Z"}}
import os, re, time, json
# import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import sys
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
from matplotlib import pyplot as plt
import pdb

# import tensorflow_datasets as tfds

print("Tensorflow version " + tf.__version__)

# %% [markdown] {"id":"HuG_q_1jkaZ6"}
# ## Parameters

# %% [markdown] {"id":"v4ocPhg6J_xw"}
# - Define the batch size
# - Define the class (category) names

# %% [code] {"id":"cCpkS9C_H7Tl","execution":{"iopub.status.busy":"2021-08-02T17:27:56.551738Z","iopub.execute_input":"2021-08-02T17:27:56.552154Z","iopub.status.idle":"2021-08-02T17:27:56.557542Z","shell.execute_reply.started":"2021-08-02T17:27:56.552093Z","shell.execute_reply":"2021-08-02T17:27:56.55645Z"}}
BATCH_SIZE = 32
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# %% [markdown] {"id":"O-o96NnyJ_xx"}
# Define some functions that will help us to create some visualizations.

# %% [code] {"id":"CfFqJxrzoj5Q","execution":{"iopub.status.busy":"2021-08-02T17:28:01.661055Z","iopub.execute_input":"2021-08-02T17:28:01.661793Z","iopub.status.idle":"2021-08-02T17:28:01.681885Z","shell.execute_reply.started":"2021-08-02T17:28:01.661731Z","shell.execute_reply":"2021-08-02T17:28:01.680314Z"}}
# Matplotlib config
# plt.rc('image', cmap='gray')
# plt.rc('grid', linewidth=0)
# plt.rc('xtick', top=False, bottom=False, labelsize='large')
# plt.rc('ytick', left=False, right=False, labelsize='large')
# plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
# plt.rc('text', color='a8151a')
# plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts
# MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

# utility to display a row of digits with their predictions
# def display_images(digits, predictions, labels, title):

#   n = 10

#   indexes = np.random.choice(len(predictions), size=n)
#   n_digits = digits[indexes]
#   n_predictions = predictions[indexes]
#   n_predictions = n_predictions.reshape((n,))
#   n_labels = labels[indexes]

#   fig = plt.figure(figsize=(20, 4))
#   plt.title(title)
#   plt.yticks([])
#   plt.xticks([])

#   for i in range(10):
#     ax = fig.add_subplot(1, 10, i+1)
#     class_index = n_predictions[i]

#     plt.xlabel(classes[class_index])
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(n_digits[i])

# # utility to display training and validation curves
# def plot_metrics(metric_name, title, ylim=5):
#   plt.title(title)
#   plt.ylim(0,ylim)
#   plt.plot(history.history[metric_name],color='blue',label=metric_name)
#   plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)

# %% [markdown] {"id":"wPq4Sw5akosT"}
# ## Loading and Preprocessing Data
# [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset has 32 x 32 RGB images belonging to 10 classes. We will load the dataset from Keras.

# %% [code] {"id":"E103YDdQ8NNq","execution":{"iopub.status.busy":"2021-08-02T17:31:58.167062Z","iopub.execute_input":"2021-08-02T17:31:58.167444Z","iopub.status.idle":"2021-08-02T17:32:18.257435Z","shell.execute_reply.started":"2021-08-02T17:31:58.167411Z","shell.execute_reply":"2021-08-02T17:32:18.254565Z"}}
def bad_res102(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    return dwnsmp

new_res = int(sys.argv[1]) if len(sys.argv) > 1 else 32
print('-----------setting resolution to {} ------'.format( new_res))
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

training_images = np.array([bad_res102(xx,(new_res,new_res)) for xx in training_images])

validation_images = training_images[-5000:]
validation_labels = training_labels[-5000:]

training_images = training_images[:-5000]
training_labels = training_labels[:-5000]

# %% [markdown] {"id":"prd944ThNavt"}
# ### Visualize Dataset
#
# Use the `display_image` to view some of the images and their class labels.

# %% [code] {"id":"UiokWTuKo88c"}
# display_images(training_images, training_labels, training_labels, "Training Data" )

# %% [code] {"id":"-q35q41KNfxH"}
# display_images(validation_images, validation_labels, validation_labels, "Training Data" )

# %% [markdown] {"id":"ltKfwrCVNuIu"}
# ### Preprocess Dataset
# Here, we'll perform normalization on images in training and validation set.
# - We'll use the function [preprocess_input](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py) from the ResNet50 model in Keras.

# %% [code] {"id":"JIxdiJVKArC6"}
def preprocess_image_input(input_images):
    input_images = input_images.astype('float32')
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims


# %% [code] {"id":"QOqjKzgAEU-Z"}
train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)

# %% [markdown] {"id":"2fooPL9Gkuox"}
# ## Define the Network
# We will be performing transfer learning on **ResNet50** available in Keras.
# - We'll load pre-trained **imagenet weights** to the model.
# - We'll choose to retain all layers of **ResNet50** along with the final classification layers.

# %% [code] {"id":"56y8UNFQIVwj"}
'''
Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
Input size is 224 x 224.
'''


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
    resize = tf.keras.layers.UpSampling2D(size=(224//new_res, 224//new_res))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output


'''
Define the model and compile it. 
Use Stochastic Gradient Descent as the optimizer.
Use Sparse Categorical CrossEntropy as the loss function.
'''


def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(new_res, new_res, 3))

    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = define_compile_model()

model.summary()

# %% [markdown] {"id":"CuhDh8ao8VyB"}
# ## Train the model

# %% [code] {"id":"2K6RNDqtJ_xx"}
pdb.set_trace()
EPOCHS = 10
history = model.fit(train_X, training_labels, epochs=EPOCHS, validation_data=(valid_X, validation_labels),
                    batch_size=64, verbose=2)

# %% [markdown] {"id":"CYb5sAEmk4ut"}
# ## Evaluate the Model
#
# Calculate the loss and accuracy metrics using the model's `.evaluate` function.

# %% [code] {"id":"io7Fuu-w3PZi"}
loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=64)

# %% [markdown] {"id":"yml-phRfPeOj"}
# ### Plot Loss and Accuracy Curves
#
# Plot the loss (in blue) and validation loss (in green).

# %% [code] {"id":"b1ZMMJ6T921A"}
# plot_metrics("loss", "Loss")

# # %% [markdown] {"id":"QbnWIbeJJ_xx"}
# # Plot the training accuracy (blue) as well as the validation accuracy (green).

# # %% [code] {"id":"P0YpFs3J99eO"}
# plot_metrics("accuracy", "Accuracy")

# # %% [markdown] {"id":"9jFVovcUUVs1"}
# # ### Visualize predictions
# # We can take a look at the predictions on the validation set.

# # %% [code] {"id":"NIQAqkMV9adq"}
# probabilities = model.predict(valid_X, batch_size=64)
# probabilities = np.argmax(probabilities, axis = 1)

# display_images(validation_images, probabilities, validation_labels, "Bad predictions indicated in red.")