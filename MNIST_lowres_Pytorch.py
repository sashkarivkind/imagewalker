
from __future__ import division, print_function, absolute_import
import numpy as np

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt


import cv2
import misc
from RL_networks import Stand_alone_net

import pickle

import importlib
importlib.reload(misc)




# PyTorch libraries and modules
from torch.utils.data import Dataset, DataLoader
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
padding_size=(16,16)
num_input = padding_size[0]*padding_size[1] # MNIST data input (img shape: 28*28)
num_classes = None 
# dropout = 0.25 # Dropout, probability to drop a unit

import matplotlib.pyplot as plt
#%%

plt.figure()
plt.imshow(np.reshape(images[0],[28,28]))

def bad_res101(batch,res):
    dwnsmp=cv2.resize(1./256*np.reshape(batch,[batch.shape[0],28,28]),res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,(28,28), interpolation = cv2.INTER_CUBIC)
    return upsmp
    

dwnsmp=cv2.resize(1./256*np.reshape(images[0],[28,28]),(10,10), interpolation = cv2.INTER_CUBIC)
upsmp = cv2.resize(dwnsmp,(28,28), interpolation = cv2.INTER_CUBIC)
plt.figure()
plt.imshow(dwnsmp)
plt.figure()
plt.imshow(upsmp)
#%%
img=1./256*np.reshape(images[0],[28,28])
for q in range(25):
    plt.figure()
    plt.imshow(bad_res101(img,(28-q,28-q)))
    
#%%
class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,4,3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4,16,3,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,4,3,stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4)
        
        self.pool = nn.MaxPool2d(2)
        
        #After the layers and pooling the first two we should get 
        # 16,3,3
        #Flatting it we get:
        # 144
        
        self.fc1 = nn.Linear(7*7*4,64)
        self.fc2 = nn.Linear(64,10)
        
        self.relu = nn.ReLU()
        
    def forward(self, img):
        
        img = self.pool(self.relu(self.bn1(self.conv1(img.double()))))
        img = self.pool(self.relu(self.bn2(self.conv2(img))))
        img = self.relu(self.bn3(self.conv3(img)))
        img = img.view(img.shape[0],7*7*4)
        
        img = self.relu(self.fc1(img))
        img = self.fc2(img)
        
        return img
#%%

images = 1./256*np.reshape(images,[-1,28,28])
labels=np.array(labels)
train_images = images[:validation_index]
validation_images = images[validation_index:]

train_labels = labels[:validation_index]
validation_labels = labels[validation_index:]


train_x = train_images.reshape(-1, 1, 28, 28)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_labels.astype(int);
train_y = torch.from_numpy(train_y)


val_x = validation_images.reshape(-1, 1, 28, 28)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = validation_labels.astype(int);
val_y = torch.from_numpy(val_y)

class mnist_dataset(Dataset):
    def __init__(self, data, labels, transform = None):
        
        self.data = data
        self.labels = labels
        
        self.transform = transform
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        args idx (int) :  index
        
        returns: tuple(trajectory, label)
        '''
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            if self.cat:
                cat_data = torch.empty
                for transform in self.transform:    
                    data1 = self.transform(data)
            data = self.transform(data)
            
            return data, label
        else:
            return data, label
    
    def dataset(self):
        return self.data
    def labels(self):
        return self.labels
   

train_dataset = mnist_dataset(train_x, train_y, 
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.Resize([10,10]),
                     torchvision.transforms.Resize([28,28])]))
    
train_dataset2 = mnist_dataset(train_x, train_y)
test_dataset = mnist_dataset(val_x, val_y)
batch = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 5_000, shuffle = True)

#%%
epochs = 51
lr = 3e-3
net = classifier().double()
optimizer = optim.Adam(net.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 5_000, shuffle = True)

train_loss = []
test_loss = []
test_accur = []
for epoch in range(epochs):
    
    batch_loss = []
    for batch_idx, (data,targets) in enumerate(train_dataloader):
        break
        optimizer.zero_grad()
        output = net(data.double())
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        
    break
    train_loss.append(np.mean(batch_loss))
        
    if epoch%5 == 0:
        correct = 0
        test_batch_loss = []
        for batch_idx, (data,targets) in enumerate(test_dataloader):
            output = net(data)
            loss = loss_func(output, targets)
            test_batch_loss.append(loss.item())
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
            test_accuracy = 100.*correct/len(targets)
            print('Epoch : ',epoch+1, '\t', 'loss :', loss.item(), 'accuracy :',test_accuracy )
        test_loss.append(np.mean(test_batch_loss))
        test_accur.append(test_accuracy)

#%%
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(196, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x.float()).float()
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if False:#torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
print(model)

def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if False:# torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train.double())
    output_val = model(x_val.double())

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)
    
    #Computing training and validation accuracies
    pred_train = output_train.data.max(1, keepdim = True)[1]
    pred_test  = output_val.data.max(1, keepdim = True)[1]
    accuracy_train = 100.*(pred_train.eq(y_train.data.view_as(pred_train)).sum())/len(y_train)
    accuracy_test  = 100.*(pred_test.eq(y_val.data.view_as(pred_test)).sum())/len(y_val)
    train_accuracies.append(accuracy_train)
    test_accuracies.append(accuracy_test)
    
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val, 'accuracy :',accuracy_test )


images = 1./256*np.reshape(images,[-1,28,28])
labels=np.array(labels)
train_images = images[:validation_index]
validation_images = images[validation_index:]

train_labels = labels[:validation_index]
validation_labels = labels[validation_index:]


train_x = train_images.reshape(-1, 1, 28, 28)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_labels.astype(int);
train_y = torch.from_numpy(train_y)


val_x = validation_images.reshape(-1, 1, 28, 28)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = validation_labels.astype(int);
val_y = torch.from_numpy(val_y)


batch_size=512
train_set_stop=-1
n_epochs = 100
#%%
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# empty list to store accuracies 
train_accuracies = []
test_accuracies = []
# training the model
for epoch in range(n_epochs):
    train(epoch)

### Now to find the effects of lowering the resolution

for epoch in range(n_epochs):
    train(epoch)

