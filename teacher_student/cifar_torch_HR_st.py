'''

out.861848
'''
import numpy as np
import pandas as pd
import sys
import os 
import matplotlib.pyplot as plt
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets

import torch.nn.functional as F

import sys

lr = float(sys.argv[1])
epochs = int(sys.argv[2])
student_fst_learning = int(sys.argv[3]) #The first learning stage of the student - number of epochs
alpha = float(sys.argv[4])
temperature = int(sys.argv[5])
beta = float(sys.argv[6])


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





def reg_training(net, lr = 1e-3, epochs = 15):
    if torch.cuda.is_available():
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
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
            output = net(HR_data.double())
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
        ###### TEST ########
        correct = 0
        test_batch_loss = []
        test_accuracy = []
        for batch_idx, (test_HR_data,test_targets) in enumerate(test_dataloader):
            if torch.cuda.is_available():
                test_HR_data = test_HR_data.to('cuda', non_blocking=True)
                test_targets = test_targets.to('cuda', non_blocking = True)
            test_output = net(test_HR_data)
            test_features = test_output[1]
            test_output = test_output[0]
            loss = loss_func(test_output, test_targets)
            test_batch_loss.append(loss.item())
            test_pred = test_output.data.max(1, keepdim = True)[1]
            correct += test_pred.eq(test_targets.data.view_as(test_pred)).sum()
        test_accur.append(100.*correct.to('cpu')/len(cifar_test))
        test_loss.append(np.mean(test_batch_loss))
        if epoch%5 == 0:
            print(#'Net',net.__class__.__name__,
              'Epoch : ',epoch+1, 
              'loss :', np.round(loss.to('cpu').item(), decimals = 4), 
              'test accuracy :',np.round(test_accur[-1].numpy(), decimals = 2),
              'train accuracy :',np.round(train_accur[-1].numpy(), decimals = 2),
                 )
    return train_accur, test_accur

def feature_learning(student_network, teacher_network, lr , epochs, beta):
    if torch.cuda.is_available():
        student_network = student_network.cuda()
        teacher_network = teacher_network.cuda()
    params = list(student_network.parameters())
    optimizer = torch.optim.Adam(params, lr = lr)
    alpha = 1 #weight of classification loss
    beta = beta #weight of feature MSE loss
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
            student_test_output, student_test_features = student_network(test_HR_data)
            teacher_test_output, teacher_test_features = teacher_network(test_HR_data)
            
            test_classification_loss = classification_loss_func(student_test_output,test_targets)
            test_feature_loss = features_loss_func(student_test_features,teacher_test_features)
            #test_loss = test_classification_loss + test_feature_loss
            
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
            student_output, student_features = student_network(HR_data.double())
            teacher_output, teacher_features = teacher_network(HR_data.double())
            
            classification_loss = classification_loss_func(student_output, targets)
            features_loss = features_loss_func(teacher_features, student_features)
            loss = beta * classification_loss + (1-beta) * features_loss
            
            loss.backward()
            optimizer.step()
            
            batch_class_loss.append(classification_loss.item())
            batch_feature_loss.append(features_loss.item())
            pred = student_output.data.max(1, keepdim = True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
        st_train_accur.append(100.*correct.to('cpu')/len(cifar_data))
        st_train_class_loss.append(np.mean(batch_class_loss))
        st_train_feature_loss.append(np.mean(batch_feature_loss))
        if epoch%10 == 0:
            print(#'Net',student_network.__class__.__name__,
                  'Epoch : ',epoch+1,
                  'class loss :', np.round(st_test_class_loss[-1], decimals = 3) ,
                  'feature loss :', np.round(st_test_features_loss[-1], decimals = 3),
                  'test accuracy :',np.round(st_test_accur[-1].numpy(), decimals = 3),
                  'train accuracy :', np.round(st_train_accur[-1].numpy(), decimals = 3),
                )
            
    return st_train_accur, st_test_accur, st_test_class_loss, st_train_feature_loss
    
def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params['alpha']
    T = params['temperature']
    KLD = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1))
    cross_ent = F.cross_entropy(outputs, labels)
    KD_loss = KLD * (alpha * T * T) + \
              cross_ent * (1. - alpha)

    return KD_loss, KLD, cross_ent
params = {
        "alpha": alpha,
    "temperature": temperature,
}

def KD(student_network, teacher_network, lr , epochs):
    if torch.cuda.is_available():
        student_network = student_network.cuda()
        teacher_network = teacher_network.cuda()
    parameters = list(student_network.parameters())
    optimizer = torch.optim.Adam(parameters, lr = lr)
    loss_func = loss_fn_kd
    features_loss_func = nn.MSELoss()
    KD_train_loss = []
    KD_train_feature_loss = []
    KD_train_accur = []
    KD_test_loss = []
    KD_train_KLD = []
    KD_test_KLD = []
    KD_test_features_loss = []
    KD_test_accur = []
    for epoch in range(epochs):
        test_batch_loss = []
        test_batch_feature_loss = []
        test_batch_KLD_loss = []
        test_correct = 0
        for batch_idx, (test_data, test_targets) in enumerate(test_dataloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                test_HR_data = test_data.cuda()
                test_targets = test_targets.cuda()
            else:
                test_HR_data = test_data
            student_test_output, student_test_features = student_network(test_HR_data)
            teacher_test_output, teacher_test_features = teacher_network(test_HR_data)
            
            test_kd_loss, test_KLD, test_cross_ent = loss_func(student_test_output,test_targets,teacher_test_output, params)
            test_feature_loss = features_loss_func(student_test_features,teacher_test_features)
            
            test_batch_loss.append(test_kd_loss.item())
            test_batch_feature_loss.append(test_feature_loss.item())
            test_batch_KLD_loss.append(test_KLD.item())
            
            test_pred = student_test_output.data.max(1, keepdim = True)[1]
            test_correct += test_pred.eq(test_targets.data.view_as(test_pred)).sum()
        KD_test_accur.append(100.*test_correct.to('cpu')/len(cifar_test))
        KD_test_loss.append(np.mean(test_batch_loss))
        KD_test_features_loss.append(np.mean(test_batch_feature_loss))
        KD_test_KLD.append(np.mean(test_batch_KLD_loss))
        batch_loss = []
        batch_feature_loss = []
        batch_KLD_loss = []
        correct = 0
        for batch_idx, (data,targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                HR_data = data.cuda()
                targets = targets.cuda()
            else:
                HR_data = data
            student_output, student_features = student_network(HR_data.double())
            teacher_output, teacher_features = teacher_network(HR_data.double())
            
            loss, train_KLD, train_cross_ent = loss_func(student_output, targets, teacher_output, params)
            features_loss = features_loss_func(teacher_features, student_features)
            
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
            batch_feature_loss.append(features_loss.item())
            batch_KLD_loss.append(train_KLD.item())
            pred = student_output.data.max(1, keepdim = True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
        KD_train_accur.append(100.*correct.to('cpu')/len(cifar_data))
        KD_train_loss.append(np.mean(batch_loss))
        KD_train_feature_loss.append(np.mean(batch_feature_loss))
        KD_train_KLD.append(np.mean(batch_KLD_loss))
        if epoch%10 == 0:
            print(#'Net',student_network.__class__.__name__,
                  'Epoch : ',epoch+1,
                  'class loss :', np.round(KD_test_loss[-1], decimals = 3) ,
                  'KLD loss :', np.round(KD_train_KLD[-1], decimals = 3),
                  'test accuracy :',np.round(KD_test_accur[-1].numpy(), decimals = 3),
                  'train accuracy :', np.round(KD_train_accur[-1].numpy(), decimals = 3),
                )
    return KD_train_accur, KD_test_accur, KD_test_loss, KD_train_feature_loss, KD_train_KLD, KD_test_KLD
    
    
    
student_teacher_data = pd.DataFrame()
student_teacher_loss = pd.DataFrame()
KD_KLD_loss = pd.DataFrame()

print('############## Training HD Teacher ###################################')
teacher_net = teacher2().double()
teacher_train_accur, teacher_test_accur = reg_training(teacher_net, lr, epochs) 
student_teacher_data['teacher_train'] = teacher_train_accur
student_teacher_data['teacher_test'] = teacher_test_accur


print('############## Training student net on HR to Get Baseline ###################################')
baseline_student = student().double()
baseline_train_accur, baseline_test_accur = reg_training(baseline_student, lr ,epochs)
student_teacher_data['baseline_train'] = baseline_train_accur
student_teacher_data['baseline_test'] = baseline_test_accur

#%%
print('Learning the Features from the teacher using MSE - no pre-train')
print('############## Learning the teachers features on HR images ###################################')
feature_student = student().double()
feature_st_train, feature_st_test, st_class, st_features = \
    feature_learning(feature_student, teacher_net, lr, epochs, beta)
st_train = feature_st_train
st_test = feature_st_test
student_teacher_data['st_train'] = st_train
student_teacher_data['st_test'] = st_test

student_teacher_loss['st_class'] = st_class
student_teacher_loss['st_feature'] = st_features


print('############## Learning the teachers softmax layer on HR images (Classic KD) #############################')
KD_student = student().double()

KD_train, KD_test, KD_class, KD_features, KD_KLD_train, KD_KLD_test = \
    KD(KD_student, teacher_net, lr, epochs)

KD_train = KD_train
KD_test = KD_test
student_teacher_data['KD_train'] = KD_train
student_teacher_data['KD_test'] = KD_test

student_teacher_loss['KD_features'] = KD_features
student_teacher_loss['KD_class'] = KD_class

KD_KLD_loss['train'] = KD_KLD_train
KD_KLD_loss['test'] = KD_KLD_test
#%%
print('Learning the Features from the teacher using MSE - WT pre Train')
print('############## Student Learning step 1 - Training student net alone  ###################################')
student_net = student().double()
student_learning_train, student_learning_test = reg_training(student_net, lr, student_fst_learning)
import copy
student_baseline = copy.deepcopy(student_net)
##################### Trying to combine loss ##########################
print('############## Learning the teachers features on HR images ###################################')
feature_student_pt = copy.deepcopy(student_net)
print(sum(sum(feature_student_pt.conv1.weight == student_net.conv1.weight)))
feature_st_train_pt, feature_st_test_pt, st_class_pt, st_features_pt = \
    feature_learning(feature_student_pt, teacher_net, lr, epochs - student_fst_learning, beta)
print(sum(sum(feature_student_pt.conv1.weight == student_net.conv1.weight)))
st_train_pt = student_learning_train + feature_st_train_pt
st_test_pt = student_learning_test + feature_st_test_pt
student_teacher_data['st_train_pt'] = st_train_pt
student_teacher_data['st_test_pt'] = st_test_pt

st_class_pt_ = np.ones(len(st_class))
st_class_pt_[:len(st_class_pt)] = st_class_pt
st_features_pt_ = np.ones(len(st_features))
st_features_pt[:len(st_class_pt)] = st_features_pt
student_teacher_loss['st_class_pt'] = st_class_pt_
student_teacher_loss['st_feature_pt'] = st_features_pt_


print('############## Learning the teachers softmax layer on HR images (Classic KD) #############################')
KD_student_pt = copy.deepcopy(student_net)
print(sum(sum(KD_student_pt.conv1.weight == student_net.conv1.weight)))
KD_train_pt, KD_test_pt, KD_class_pt, KD_features_pt, KD_KLD_train_pt, KD_KLD_test_pt = \
    KD(KD_student_pt, teacher_net, lr, epochs - student_fst_learning)
print(sum(sum(KD_student_pt.conv1.weight == student_net.conv1.weight)))
KD_train_pt = student_learning_train + KD_train_pt
KD_test_pt = student_learning_test + KD_test_pt
student_teacher_data['KD_train_pt'] = KD_train_pt
student_teacher_data['KD_test_pt'] = KD_test_pt


KD_features_pt_ = np.ones(len(KD_features))
KD_features_pt_[:len(KD_features_pt)] = KD_features_pt
KD_class_pt_ = np.ones(len(KD_class))
KD_class_pt_[:len(KD_class_pt)] = KD_class_pt
student_teacher_loss['KD_features_pt'] = KD_features_pt_
student_teacher_loss['KD_class_pt'] = KD_class_pt_

KD_KLD_train_ = np.ones(len(KD_KLD_train))
KD_KLD_train_[:len(KD_KLD_train_pt)] = KD_KLD_train_pt
KD_KLD_test_ = np.ones(len(KD_KLD_test))
KD_KLD_test_[:len(KD_KLD_train_pt)] = KD_KLD_test_pt
KD_KLD_loss['train pt'] = KD_KLD_train_
KD_KLD_loss['test pt'] = KD_KLD_test_
#%%
############################ SAVE AND PLOT ###################################
##############################################################################

student_teacher_data.to_pickle('cifar_HR_ST_data_{:.0e}_{}_{}.pkl'.format(lr,epochs,beta))
student_teacher_loss.to_pickle('cifar_HR_ST_loss_{:.0e}_{}_{}.pkl'.format(lr,epochs,beta))
KD_KLD_loss.to_pickle('cifar_HR_KLD_loss_{:.0e}_{}_{}.pkl'.format(lr,epochs,beta))

plt.figure()
#plt.plot(teacher_train_accur, label = 'teacher train')
plt.plot(teacher_test_accur, label = 'teacher test')
plt.plot(baseline_test_accur, label = 'baseline test')
#plt.plot(baseline_train_accur, label = 'baseline train')
plt.plot(st_test, label = 'st test')
#plt.plot(st_train, label = 'st train')
plt.plot(KD_test, label = 'KD test')
#plt.plot(KD_train, label = 'KD train')
plt.plot(st_test_pt, label = 'st test pt')
plt.plot(KD_test_pt, label = 'KD test pt')
plt.legend()
plt.title('Student Teacher Cifar Accuracy')
plt.savefig('teacher_student_cifar_accuracy_{:.0e}_{}_{}_{}.png'.format(lr,epochs,beta,student_fst_learning))

plt.figure()
plt.plot(st_features, label = 'st features')
plt.plot(KD_features, label = 'KD features')
plt.plot(st_features_pt, label = 'st features pt')
plt.plot(KD_features_pt, label = 'KD features pt')
plt.legend()
plt.title('Student Teacher Cifar feature Loss')
plt.savefig('teacher_student_cifar_feature_loss_{:.0e}_{}_{}_{}'.format(lr,epochs,beta,student_fst_learning))

plt.figure()
plt.plot(st_class, label = 'st class')
plt.plot(KD_class, label = 'KD class')
plt.plot(st_class_pt, label = 'st class pt')
plt.plot(KD_class_pt, label = 'KD class pt')
plt.legend()
plt.title('Student Teacher Cifar Class Loss')
plt.savefig('teacher_student_cifar_class_loss_{:.0e}_{}_{}_{}'.format(lr,epochs,beta,student_fst_learning))

