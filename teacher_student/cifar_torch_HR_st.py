'''

out.861848
'''
import numpy as np
import pandas as pd
import sys
import os 
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_data = datasets.CIFAR10('/home/labs/ahissarlab/orra/datasets/',download = True,
                            
                            transform = transform)

train_dataloader = torch.utils.data.DataLoader(cifar_data,
                                          batch_size=64,
                                          shuffle=True,
                                           )

cifar_test = datasets.CIFAR10('/home/labs/ahissarlab/orra/datasets/', download = True,
                            train = False,transform = transform)

test_dataloader = torch.utils.data.DataLoader(cifar_test,
                                          batch_size=64,
                                          shuffle=True,
                                         
                                          )

from functools import reduce
from operator import __add__

class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)

class teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,stride=1, padding = 1)
        self.conv12 = nn.Conv2d(32,32,3,stride=1, padding = 1)
        self.drop1 = nn.Dropout(p = 0.2)
        self.conv2 = nn.Conv2d(32,64,3,stride=1, padding = 1)
        self.conv22 = nn.Conv2d(64,64,3,stride=1, padding = 1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.conv3 = nn.Conv2d(64,128,3,stride=1, padding = 1)
        self.conv32 = nn.Conv2d(128,128,3,stride=1, padding = 1)
        self.drop3 = nn.Dropout(p = 0.2)
        
        self.pool = nn.MaxPool2d(2)
        #After the layers and pooling the first two we should get 
        # 16,3,3
        #Flatting it we get:
        # 144
        
        self.fc1 = nn.Linear(4*4*128,128)
        self.fc2 = nn.Linear(128,10)
        
        self.relu = nn.ReLU()
        
    def forward(self, img):
        
        img = self.relu(self.conv1(img.double()))
        img = self.drop1(self.pool(self.relu(self.conv12(img))))
        img = self.relu(self.conv2(img))
        img = self.drop2(self.pool(self.relu(self.conv22(img))))
        img = self.relu(self.conv3(img))
        img = self.drop3(self.pool(self.relu(self.conv32(img))))       
        #print(img.shape)
        img_features = img.view(img.shape[0],4*4*128)
        img = self.relu(self.fc1(img_features))
        img = self.fc2(img)
        
        return img, img_features

class teacher2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,stride=1, padding = 1)
        self.conv12 = nn.Conv2d(32,32,3,stride=1)
        self.drop1 = nn.Dropout(p = 0.2)
        self.conv2 = nn.Conv2d(32,64,3,stride=1, padding = 1)
        self.conv22 = nn.Conv2d(64,64,3,stride=1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.conv3 = nn.Conv2d(64,128,3,stride=1, padding = 1)
        self.conv32 = nn.Conv2d(128,128,3,stride=1)
        self.drop3 = nn.Dropout(p = 0.2)
        
        self.pool = nn.MaxPool2d(2)
        #After the layers and pooling the first two we should get 
        # 16,3,3
        #Flatting it we get:
        # 144
        
        self.fc1 = nn.Linear(2*2*128,128)
        self.fc2 = nn.Linear(128,10)
        
        self.relu = nn.ReLU()
        
    def forward(self, img):
        
        img = self.relu(self.conv1(img.double()))
        img = self.drop1(self.pool(self.relu(self.conv12(img))))
        img = self.relu(self.conv2(img))
        img = self.drop2(self.pool(self.relu(self.conv22(img))))
        img = self.relu(self.conv3(img))
        img = self.drop3(self.pool(self.relu(self.conv32(img))))       
        #print(img.shape)
        img_features = img.view(img.shape[0],2*2*128)
        img = self.relu(self.fc1(img_features))
        img = self.fc2(img)
        
        return img, img_features
    
class student(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,3,stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2)
        
        #After the layers and pooling the first two we should get 
        # 16,3,3
        #Flatting it we get:
        # 144
        
        self.fc1 = nn.Linear(4*4*32,64)
        self.fc2 = nn.Linear(64,10)
        
        self.relu = nn.ReLU()
        
    def forward(self, img):
        
        img = self.pool(self.bn1(self.relu(self.conv1(img.double()))))
        img = self.pool(self.bn2(self.relu(self.conv2(img))))
        img = self.pool(self.bn3(self.relu(self.conv3(img))))       
        #print(img.shape)
        img_features = img.view(img.shape[0],4*4*32)
        img = self.relu(self.fc1(img_features))
        img = self.fc2(img)
        
        return img,img_features    


from functools import reduce

print('############## Training HD Teacher ###################################')
lr = 1e-3
epochs = 15
teacher_net = teacher2().double()
if torch.cuda.is_available():
    teacher_net.cuda()
optimizer = torch.optim.Adam(teacher_net.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()
train_loss = []
train_accur = []
test_loss = []
test_accur = []
for epoch in range(epochs):
    batch_loss = []
    train_accuracy = []
    correct = 0
    for batch_idx, (HR_data, targets) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            HR_data = HR_data.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking = True)
        optimizer.zero_grad()
        output = teacher_net(HR_data.double())
        features = output[1]
        output = output[0]
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
    train_accur.append(100.*correct.to('cpu')/len(cifar_data))
    train_loss.append(np.mean(batch_loss))
    correct = 0
    test_batch_loss = []
    test_accuracy = []
    
    for batch_idx, (test_HR_data,test_targets) in enumerate(test_dataloader):
        if torch.cuda.is_available():
            test_HR_data = HR_data.to('cuda', non_blocking=True)
            test_targets = targets.to('cuda', non_blocking = True)
        test_output = teacher_net(test_HR_data)
        test_features = test_output[1]
        test_output = test_output[0]
        loss = loss_func(test_output, test_targets)
        test_batch_loss.append(loss.item())
        test_pred = test_output.data.max(1, keepdim = True)[1]
        correct += test_pred.eq(test_targets.data.view_as(test_pred)).sum()
    test_accur.append(100.*correct.to('cpu')/len(cifar_test))
    test_loss.append(np.mean(test_batch_loss))
    print('Net',teacher_net.__class__.__name__,
      'Epoch : ',epoch+1, 
      'loss :', np.round(loss.to('cpu').item(), decimals = 4), 
      'test accuracy :',np.round(test_accur[-1].numpy(), decimals = 2),
      'train accuracy :',np.round(train_accur[-1].numpy(), decimals = 2),
         )




print('############## Training student net on HR to Get Baseline ###################################')
lr = 1e-3
epochs = 30
student_baseline = student().double()
if torch.cuda.is_available():
    student_baseline.cuda()
optimizer = torch.optim.Adam(student_baseline.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()
train_loss = []
train_accur = []
test_loss = []
test_accur = []
for epoch in range(epochs):
    batch_loss = []
    correct = 0
    for batch_idx, (HR_data, targets) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            HR_data = HR_data.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking = True)
        optimizer.zero_grad()
        output = student_baseline(HR_data.double())
        features = output[1]
        output = output[0]
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
    train_accur.append(100.*correct.to('cpu')/len(cifar_data))
    train_loss.append(np.mean(batch_loss))
    correct = 0
    test_batch_loss = []
    for batch_idx, (test_HR_data,test_targets) in enumerate(test_dataloader):
        if torch.cuda.is_available():
            test_HR_data = test_HR_data.to('cuda', non_blocking=True)
            test_targets = targets.to('cuda', non_blocking = True)
        test_output = student_baseline(test_HR_data)
        test_features = test_output[1]
        test_output = test_output[0]
        loss = loss_func(test_output, test_targets)
        test_batch_loss.append(loss.item())
        test_pred = test_output.data.max(1, keepdim = True)[1]
        correct += test_pred.eq(test_targets.data.view_as(test_pred)).sum()
    test_accur.append(100.*correct.to('cpu')/len(cifar_test))
    test_loss.append(np.mean(test_batch_loss))
    print('Net',teacher_net.__class__.__name__,
          'Epoch : ',epoch+1, 
          'loss :', loss.to('cpu').item(), 
          'test accuracy :',np.round(test_accur[-1].numpy(), decimals = 2),
          'train accuracy :',np.round(train_accur[-1].numpy(), decimals = 2),
             )





baseline_test = test_accur
baseline_train = train_accur
print(baseline_test)
print(baseline_train)

plt.figure()
plt.plot(baseline_test, label = 'test')
plt.plot(baseline_train, label = 'train')
plt.legend()
plt.title('Student network Baseline')

print('############## Training student net on HR to Get Baseline ###################################')
lr = 1e-3
epochs = 10
student_baseline = student().double()
if torch.cuda.is_available():
    student_baseline.cuda()
optimizer = torch.optim.Adam(student_baseline.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()
train_loss = []
train_accur = []
test_loss = []
test_accur = []
for epoch in range(epochs):
    batch_loss = []
    correct = 0
    for batch_idx, (HR_data, targets) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            HR_data = HR_data.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking = True)
        optimizer.zero_grad()
        output = student_baseline(HR_data.double())
        features = output[1]
        output = output[0]
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
    train_accur.append(100.*correct.to('cpu')/len(cifar_data))
    train_loss.append(np.mean(batch_loss))
    correct = 0
    test_batch_loss = []
    for batch_idx, (test_HR_data,test_targets) in enumerate(test_dataloader):
        if torch.cuda.is_available():
            test_HR_data = test_HR_data.to('cuda', non_blocking=True)
            test_targets = targets.to('cuda', non_blocking = True)
        test_output = student_baseline(test_HR_data)
        test_features = test_output[1]
        test_output = test_output[0]
        loss = loss_func(test_output, test_targets)
        test_batch_loss.append(loss.item())
        test_pred = test_output.data.max(1, keepdim = True)[1]
        correct += test_pred.eq(test_targets.data.view_as(test_pred)).sum()
    test_accur.append(100.*correct.to('cpu')/len(cifar_test))
    test_loss.append(np.mean(test_batch_loss))
    print('Net',teacher_net.__class__.__name__,
          'Epoch : ',epoch+1, 
          'loss :', loss.to('cpu').item(), 
          'test accuracy :',np.round(test_accur[-1].numpy(), decimals = 2),
          'train accuracy :',np.round(train_accur[-1].numpy(), decimals = 2),
             )

##################### Trying to combine loss ##########################
print('############## Learning the teachers features on HR images ###################################')
lr = 1e-3
epochs = 20
student_net = student_baseline
if torch.cuda.is_available():
    student_net = student_net.cuda()
params = list(student_net.parameters())
optimizer = torch.optim.Adam(params, lr = lr)
alpha = 1 #weight of classification loss
beta = 10 #weight of feature MSE loss
classification_loss_func = nn.CrossEntropyLoss()
features_loss_func = nn.MSELoss()
st_train_class_loss = []
st_train_feature_loss = []
st_train_accur = []
st_test_class_loss = []
st_test_features_loss = []
st_test_accur = []
for epoch in range(epochs):
    test_batch_class_loss = []
    test_batch_feature_loss = []
    test_correct = 0
    for batch_idx, (test_data, test_targets) in enumerate(test_dataloader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            test_HR_data = test_data.cuda()
            test_targets = test_targets.cuda()
        else:
            test_HR_data = test_data
        student_test_output, student_test_features = student_net(test_HR_data)
        teacher_test_output, teacher_test_features = teacher_net(test_HR_data)
        
        test_classification_loss = classification_loss_func(student_test_output,test_targets)
        test_feature_loss = features_loss_func(student_test_features,teacher_test_features)
        test_loss = test_classification_loss + test_feature_loss
        
        test_batch_class_loss.append(test_classification_loss.item())
        test_batch_feature_loss.append(test_feature_loss.item())
        
        test_pred = student_test_output.data.max(1, keepdim = True)[1]
        test_correct += test_pred.eq(test_targets.data.view_as(test_pred)).sum()
    st_test_accur.append(100.*test_correct.to('cpu')/len(cifar_test))
    st_test_class_loss.append(np.mean(test_batch_class_loss))
    st_test_features_loss.append(np.mean(test_batch_feature_loss))
    batch_class_loss = []
    batch_feature_loss = []
    
    correct = 0
    for batch_idx, (data,targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            HR_data = data.cuda()
            targets = targets.cuda()
        else:
            HR_data = data
        student_output, student_features = student_net(HR_data.double())
        teacher_output, teacher_features = teacher_net(HR_data.double())
        
        classification_loss = classification_loss_func(student_output, targets)
        features_loss = features_loss_func(teacher_features, student_features)
        loss = alpha * classification_loss + beta * features_loss
        
        loss.backward()
        optimizer.step()
        
        batch_class_loss.append(classification_loss.item())
        batch_feature_loss.append(features_loss.item())
        pred = student_output.data.max(1, keepdim = True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
    st_train_accur.append(100.*correct.to('cpu')/len(cifar_data))
    st_train_class_loss.append(np.mean(batch_class_loss))
    st_train_feature_loss.append(np.mean(batch_feature_loss))

    print('Net',student_net.__class__.__name__,
          'Epoch : ',epoch+1,
          'class loss :', np.round(st_test_class_loss[-1], decimals = 3) ,
          'feature loss :', np.round(st_test_features_loss[-1], decimals = 3),
          'test accuracy :',np.round(st_test_accur[-1].numpy(), decimals = 3),
          'train accuracy :', np.round(st_train_accur[-1].numpy(), decimals = 3),
        )
    
        
combined_test = test_accur + st_test_accur
combined_train = train_accur + st_train_accur

print(combined_test)
print(combined_train)

print('############## Training student net on HR to Get Baseline ###################################')
lr = 1e-3
epochs = 10
student_baseline = student().double()
if torch.cuda.is_available():
    student_baseline.cuda()
optimizer = torch.optim.Adam(student_baseline.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()
train_loss = []
train_accur = []
test_loss = []
test_accur = []
for epoch in range(epochs):
    batch_loss = []
    correct = 0
    for batch_idx, (HR_data, targets) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            HR_data = HR_data.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking = True)
        optimizer.zero_grad()
        output = student_baseline(HR_data.double())
        features = output[1]
        output = output[0]
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
    train_accur.append(100.*correct.to('cpu')/len(cifar_data))
    train_loss.append(np.mean(batch_loss))
    correct = 0
    test_batch_loss = []
    for batch_idx, (test_HR_data,test_targets) in enumerate(test_dataloader):
        if torch.cuda.is_available():
            test_HR_data = test_HR_data.to('cuda', non_blocking=True)
            test_targets = targets.to('cuda', non_blocking = True)
        test_output = student_baseline(test_HR_data)
        test_features = test_output[1]
        test_output = test_output[0]
        loss = loss_func(test_output, test_targets)
        test_batch_loss.append(loss.item())
        test_pred = test_output.data.max(1, keepdim = True)[1]
        correct += test_pred.eq(test_targets.data.view_as(test_pred)).sum()
    test_accur.append(100.*correct.to('cpu')/len(cifar_test))
    test_loss.append(np.mean(test_batch_loss))
    print('Net',teacher_net.__class__.__name__,
          'Epoch : ',epoch+1, 
          'loss :', loss.to('cpu').item(), 
          'test accuracy :',np.round(test_accur[-1].numpy(), decimals = 2),
          'train accuracy :',np.round(train_accur[-1].numpy(), decimals = 2),
             )


print('############## Learning the teachers softmax layer on HR images (Classic KD) #############################')
import torch.nn.functional as F
def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params['alpha']
    T = params['temperature']
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss
params = {
        "alpha": 0.9,
    "temperature": 20,
}

lr = 1e-3
epochs = 40
student_net = student_baseline
if torch.cuda.is_available():
    student_net = student_net.cuda()
parameters = list(student_net.parameters())
optimizer = torch.optim.Adam(parameters, lr = lr)
loss_func = loss_fn_kd
features_loss_func = nn.MSELoss()
train_loss = []
train_feature_loss = []
train_accur = []
test_loss = []
test_features_loss = []
test_accur = []
for epoch in range(epochs):
    test_batch_loss = []
    test_batch_feature_loss = []
    test_correct = 0
    for batch_idx, (test_data, test_targets) in enumerate(test_dataloader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            test_HR_data = test_data.cuda()
            test_targets = test_targets.cuda()
        else:
            test_HR_data = test_data
        student_test_output, student_test_features = student_net(test_HR_data)
        teacher_test_output, teacher_test_features = teacher_net(test_HR_data)
        
        test_kd_loss = loss_func(student_test_output,test_targets,teacher_test_output, params)
        test_feature_loss = features_loss_func(student_test_features,teacher_test_features)
        
        test_batch_loss.append(test_kd_loss.item())
        test_batch_feature_loss.append(test_feature_loss.item())
        
        test_pred = student_test_output.data.max(1, keepdim = True)[1]
        test_correct += test_pred.eq(test_targets.data.view_as(test_pred)).sum()
    test_accur.append(100.*test_correct.to('cpu')/len(cifar_test))
    test_loss.append(np.mean(test_batch_loss))
    test_features_loss.append(np.mean(test_batch_feature_loss))
    batch_loss = []
    batch_feature_loss = []
    
    correct = 0
    for batch_idx, (data,targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            HR_data = data.cuda()
            targets = targets.cuda()
        else:
            HR_data = data
        student_output, student_features = student_net(HR_data.double())
        teacher_output, teacher_features = teacher_net(HR_data.double())
        
        loss = loss_func(student_output, targets, teacher_output, params)
        features_loss = features_loss_func(teacher_features, student_features)
        
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        batch_feature_loss.append(features_loss.item())
        pred = student_output.data.max(1, keepdim = True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
    train_accur.append(100.*correct.to('cpu')/len(cifar_data))
    train_loss.append(np.mean(batch_loss))
    train_feature_loss.append(np.mean(batch_feature_loss))

    print('Net',student_net.__class__.__name__,
          'Epoch : ',epoch+1,
          'class loss :', np.round(test_loss[-1], decimals = 3) ,
          'feature loss :', np.round(test_features_loss[-1], decimals = 3),
          'test accuracy :',np.round(test_accur[-1].numpy(), decimals = 3),
          'train accuracy :', np.round(train_accur[-1].numpy(), decimals = 3),
        )
            
KD_combined_test = test_accur + st_test_accur
KD_combined_train = train_accur + st_train_accur

print(KD_combined_test)
print(KD_combined_train)

plt.figure()
plt.plot(baseline_test, label = 'baseline_test')
plt.plot(baseline_train, label = 'baseline_train')
plt.plot(combined_test, label = 'combined_test')
plt.plot(combined_train, label = 'combined_train')
plt.plot(KD_combined_train, label = 'KD_combined_train')
plt.plot(KD_combined_train, label = 'KD_combined_train')
plt.savefig('teacher student on cifar.png')
