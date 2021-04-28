from __future__ import division, print_function, absolute_import

import numpy as np
import cv2
import misc
from RL_networks import Stand_alone_net

import pickle

import importlib
importlib.reload(misc)




# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

from mnist import MNIST

mnist = MNIST('/home/orram/Documents/datasets/MNIST/')
images, labels = mnist.load_training()

# Training Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 128

validation_index=-5000

# Network Parameters
size=None
padding_size=(128,128)
# num_input = padding_size[0]*padding_size[1] # MNIST data input (img shape: 28*28)
num_classes = None 
# dropout = 0.25 # Dropout, probability to drop a unit

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(misc.build_mnist_padded(1./256*np.reshape(images[0],[1,28,28])))

img=misc.build_mnist_padded(1./256*np.reshape(images[0],[1,28,28]))

import SYCLOP_env as syc

def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh, interpolation = cv2.INTER_CUBIC)
    return upsmp
    

scene = syc.Scene(image_matrix=img)
sensor = syc.Sensor(winx=56,winy=56,centralwinx=28,centralwiny=28)
agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])

q_sequence = [[agent.max_q[0]//2, qq ] for qq in np.arange(agent.max_q[1]//2-10,agent.max_q[1]//2+10)]
q_sequence = np.array(q_sequence)

sensor.hp.resolution_fun = lambda x: bad_res101(x,(10,10))
imim=[]
dimim=[]
agent.set_manual_trajectory(manual_q_sequence=q_sequence)
for t in range(70):
    agent.manual_act()
    sensor.update(scene, agent)
    imim.append(sensor.central_frame_view)
    dimim.append(sensor.central_dvs_view)
    

for i in range(20):
    plt.figure()
    plt.imshow(imim[i])
'''
### Create a Dataset from the Syclops visual inputs
We are starting with a simple time series where the syclop starts from the same starting point, at the middle of the img on the x axis and the middle - 10 pixles on the y axis - (middle_point, middle_point - 10)
<br> Each time step the syclop will move one pixle up on the y axis, to a final point at (middle_point, middle_point + 10)
<br> 
'''

#Setting the coordinates to visit
q_sequence = [[agent.max_q[0]//2, qq ] for qq in np.arange(agent.max_q[1]//2-10,agent.max_q[1]//2+10,4)]
q_sequence = np.array(q_sequence)
#Setting the resolution function - starting with the regular resolution
sensor.hp.resolution_fun = lambda x: bad_res101(x,(28,28))

count = 0

ts_images = []
dvs_images = []
count = 0
for img in images[55_000:]:
    #Set the padded image
    img=misc.build_mnist_padded(1./256*np.reshape(img,[1,28,28]))
    if count < 10:
        plt.figure()
        plt.imshow(img)
    count+=1
    #Set the sensor and the agent
    scene = syc.Scene(image_matrix=img)
    sensor = syc.Sensor(winx=56,winy=56,centralwinx=28,centralwiny=28)
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])
    #Create empty lists to store the syclops outputs
    imim=[]
    dimim=[]
    agent.set_manual_trajectory(manual_q_sequence=q_sequence)
    #Run Syclop for 20 time steps
    for t in range(5):
        agent.manual_act()
        sensor.update(scene, agent)
        imim.append(sensor.central_frame_view)
        dimim.append(sensor.central_dvs_view)
    #Create a unified matrix from the list
    if count < 26:
        plt.figure()
        plt.imshow(imim[-1])
    imim = np.array(imim)
    dimim = np.array(dimim)
    #Add current proccessed image to lists
    ts_images.append(imim)
    dvs_images.append(dimim)
    count += 1
    
    



