#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Comparing student teacher methods -  using Knowledge Disillation (KD) and Feature Learning (FL)

Using Keras and Cifar 10 with syclop. 

A pervious comperision was made to distill knowledge with pytorch on cifar HR
images - there we found that the KD outperformed FL.
Another comparision was made with MNIST and pytorch with syclop on LR images - 
there we found that FL outperformed KD - the opposite!

lr = 5e-4 alpha = 0.9 beta = 0.8 temp = 10 - out.
lr = 1e-4 alpha = 0.9 beta = 0.9 temp = 10 - out.299259 pre-train start = 1 (mistake) Batch norm
KD test accuracy =  [0.38359999656677246, 0.4424000084400177, 0.503000020980835, 0.5307999849319458, 0.550599992275238, 0.5595999956130981, 0.5555999875068665, 0.5889000296592712, 0.5857999920845032, 0.6021999716758728, 0.6140999794006348, 0.6348000168800354, 0.6157000064849854, 0.637499988079071, 0.6445000171661377, 0.6340000033378601, 0.6309999823570251, 0.6347000002861023, 0.6434999704360962, 0.6614999771118164, 0.636900007724762, 0.652400016784668, 0.6646999716758728, 0.6478000283241272, 0.6377000212669373, 0.6610999703407288, 0.6563000082969666, 0.6481000185012817, 0.6621000170707703, 0.6675000190734863, 0.6732000112533569, 0.6582000255584717, 0.6676999926567078, 0.6585000157356262, 0.6664000153541565, 0.6693999767303467, 0.666100025177002, 0.6586999893188477, 0.6751999855041504, 0.675000011920929, 0.6607999801635742, 0.6448000073432922, 0.6689000129699707, 0.6642000079154968, 0.6776999831199646, 0.6704000234603882, 0.6786999702453613, 0.6700999736785889, 0.6769000291824341, 0.67330002784729, 0.6672999858856201, 0.6717000007629395, 0.6694999933242798, 0.6743999719619751, 0.6622999906539917, 0.6733999848365784, 0.6693000197410583, 0.6754000186920166, 0.6449000239372253, 0.6833000183105469, 0.6685000061988831, 0.6626999974250793, 0.6710000038146973, 0.6789000034332275, 0.6622999906539917, 0.6714000105857849, 0.6769000291824341, 0.6797000169754028, 0.6776000261306763, 0.65829998254776, 0.6639999747276306, 0.6747999787330627, 0.6567999720573425, 0.6682000160217285, 0.6603000164031982, 0.6699000000953674, 0.6675000190734863, 0.6679999828338623, 0.667900025844574, 0.6736999750137329, 0.66839998960495, 0.6743999719619751, 0.6640999913215637, 0.6693000197410583, 0.6776000261306763, 0.6714000105857849, 0.669700026512146, 0.6744999885559082, 0.666100025177002, 0.6590999960899353, 0.6784999966621399, 0.6643999814987183, 0.6771000027656555, 0.6764000058174133, 0.6697999835014343, 0.6700999736785889, 0.661899983882904, 0.6674000024795532, 0.6657999753952026, 0.670199990272522, 0.6682000160217285, 0.6704000234603882, 0.6603999733924866, 0.6718000173568726, 0.66839998960495, 0.6516000032424927, 0.6651999950408936, 0.6678000092506409, 0.6700000166893005, 0.6686000227928162, 0.6729000210762024, 0.6747000217437744, 0.6801000237464905, 0.6520000100135803, 0.6672999858856201, 0.6718000173568726, 0.6480000019073486, 0.6747999787330627, 0.6696000099182129, 0.6735000014305115, 0.6620000004768372, 0.6690000295639038, 0.6700999736785889, 0.6621999740600586, 0.6744999885559082, 0.6614000201225281, 0.6581000089645386, 0.6547999978065491, 0.6639999747276306, 0.6601999998092651, 0.6689000129699707, 0.676800012588501, 0.6739000082015991, 0.6725999712944031, 0.6735000014305115, 0.666100025177002, 0.6639000177383423, 0.6699000000953674, 0.670799970626831, 0.6717000007629395, 0.6495000123977661, 0.6711000204086304, 0.6593999862670898, 0.6692000031471252, 0.66839998960495, 0.67540001869201
KD2 test accuracy =  [0.3903999924659729, 0.4415999948978424, 0.5016999840736389, 0.5378999710083008, 0.5444999933242798, 0.5802000164985657, 0.5964000225067139, 0.5879999995231628, 0.6068999767303467, 0.6144999861717224, 0.6115000247955322, 0.6259999871253967, 0.6320000290870667, 0.6388999819755554, 0.6535999774932861, 0.6373000144958496, 0.633899986743927, 0.6557999849319458, 0.6507999897003174, 0.652899980545044, 0.654699981212616, 0.6542999744415283, 0.6624000072479248, 0.6575000286102295, 0.656000018119812, 0.6657000184059143, 0.6589000225067139, 0.6714000105857849, 0.6679999828338623, 0.676800012588501, 0.6639999747276306, 0.6754999756813049, 0.6686000227928162, 0.6740999817848206, 0.6814000010490417, 0.6712999939918518, 0.6794999837875366, 0.6819999814033508, 0.6819000244140625, 0.6662999987602234, 0.6438999772071838, 0.6700999736785889, 0.6845999956130981, 0.6812000274658203, 0.6664999723434448, 0.6887000203132629, 0.6789000034332275, 0.6940000057220459, 0.6840000152587891, 0.6780999898910522, 0.6830999851226807, 0.692300021648407, 0.6761999726295471, 0.6876000165939331, 0.6832000017166138, 0.6776999831199646, 0.6888999938964844, 0.685699999332428, 0.6807000041007996, 0.6833999752998352, 0.6883000135421753, 0.6848000288009644, 0.6779000163078308, 0.6876000165939331, 0.6818000078201294, 0.6787999868392944, 0.6887999773025513, 0.6858000159263611, 0.6906999945640564, 0.675599992275238, 0.6935999989509583, 0.6879000067710876, 0.6843000054359436, 0.6872000098228455, 0.6934999823570251, 0.6916999816894531, 0.6780999898910522, 0.6938999891281128, 0.6779999732971191, 0.682699978351593, 0.6955000162124634, 0.6929000020027161, 0.6830999851226807, 0.6948999762535095, 0.683899998664856, 0.6887000203132629, 0.6854000091552734, 0.6898999810218811, 0.6931999921798706, 0.6832000017166138, 0.6901000142097473, 0.6876000165939331, 0.6930000185966492, 0.6881999969482422, 0.6855999827384949, 0.6865000128746033, 0.6941999793052673, 0.6836000084877014, 0.6945000290870667, 0.6868000030517578, 0.6868000030517578, 0.695900022983551, 0.6818000078201294, 0.6868000030517578, 0.694100022315979, 0.6797999739646912, 0.6934999823570251, 0.6977999806404114, 0.7016000151634216, 0.6869999766349792, 0.6933000087738037, 0.6988000273704529, 0.6883999705314636, 0.7002999782562256, 0.697700023651123, 0.6995999813079834, 0.6894000172615051, 0.6942999958992004, 0.6840999722480774, 0.6898999810218811, 0.6938999891281128, 0.6958000063896179, 0.684499979019165, 0.6797999739646912, 0.6829000115394592, 0.6998000144958496, 0.6798999905586243, 0.7002999782562256, 0.689300000667572, 0.6965000033378601, 0.6866000294685364, 0.6887000203132629, 0.6952999830245972, 0.6920999884605408, 0.6899999976158142, 0.6991999745368958, 0.6935999989509583, 0.7008000016212463, 0.7019000053405762, 0.7009000182151794, 0.6869999766349792, 0.7017999887466431, 0.6956999897956848, 0.6987000107765198, 0.6962000131607056, 0.6916000247001648, 0.684499979019165, 0.6852999925613403, 0.6710000038146973, 0.7045999765396118]
Baseline test accuracy =  [0.3813999891281128, 0.44440001249313354, 0.4896000027656555, 0.5184000134468079, 0.5554999709129333, 0.5534999966621399, 0.5835999846458435, 0.5953999757766724, 0.5985999703407288, 0.6121000051498413, 0.6229000091552734, 0.6019999980926514, 0.6290000081062317, 0.6295999884605408, 0.6342999935150146, 0.6355000138282776, 0.6305000185966492, 0.6514000296592712, 0.6403999924659729, 0.656000018119812, 0.6549000144004822, 0.6552000045776367, 0.6575000286102295, 0.6589999794960022, 0.6590999960899353, 0.6550999879837036, 0.6682999730110168, 0.656000018119812, 0.6532999873161316, 0.6682999730110168, 0.6736999750137329, 0.6607999801635742, 0.6642000079154968, 0.663100004196167, 0.6704999804496765, 0.6703000068664551, 0.6751000285148621, 0.6794999837875366, 0.6730999946594238, 0.6761999726295471, 0.6754999756813049, 0.6761999726295471, 0.6776000261306763, 0.6765000224113464, 0.6700000166893005, 0.6718000173568726, 0.6625999808311462, 0.6697999835014343, 0.6581000089645386, 0.6692000031471252, 0.6740000247955322, 0.6804999709129333, 0.669700026512146, 0.6816999912261963, 0.6776000261306763, 0.6765000224113464, 0.6740000247955322, 0.6772000193595886, 0.6775000095367432, 0.676800012588501, 0.682699978351593, 0.6830999851226807, 0.6765000224113464, 0.6805999875068665, 0.6712999939918518, 0.6746000051498413, 0.6832000017166138, 0.677299976348877, 0.6708999872207642, 0.6765000224113464, 0.6829000115394592, 0.6832000017166138, 0.6796000003814697, 0.6549000144004822, 0.6696000099182129, 0.6747999787330627, 0.6759999990463257, 0.6593999862670898, 0.6732000112533569, 0.6751999855041504, 0.6714000105857849, 0.6740999817848206, 0.678600013256073, 0.6714000105857849, 0.6747999787330627, 0.6751000285148621, 0.6801999807357788, 0.6818000078201294, 0.6739000082015991, 0.676800012588501, 0.6725999712944031, 0.6722000241279602, 0.6804999709129333, 0.6827999949455261, 0.6820999979972839, 0.6711999773979187, 0.6661999821662903, 0.6628999710083008, 0.666700005531311, 0.6753000020980835, 0.6775000095367432, 0.6694999933242798, 0.6730999946594238, 0.680899977684021, 0.6694999933242798, 0.6639999747276306, 0.6636000275611877, 0.6710000038146973, 0.6740000247955322, 0.6685000061988831, 0.6722999811172485, 0.6696000099182129, 0.6696000099182129, 0.6722999811172485, 0.6747000217437744, 0.6722000241279602, 0.6739000082015991, 0.667900025844574, 0.6593999862670898, 0.6690000295639038, 0.6692000031471252, 0.6699000000953674, 0.670799970626831, 0.6707000136375427, 0.6711000204086304, 0.6692000031471252, 0.670199990272522, 0.6646000146865845, 0.6674000024795532, 0.6723999977111816, 0.6722999811172485, 0.6639000177383423, 0.6689000129699707, 0.6718000173568726, 0.647599995136261, 0.6787999868392944, 0.6632999777793884, 0.6668000221252441, 0.6675999760627747, 0.6528000235557556, 0.6735000014305115, 0.663100004196167, 0.6728000044822693, 0.6740999817848206, 0.661899983882904, 0.6662999987602234, 0.6699000000953674, 0.6496000289916992, 0.6586999893188477, 0.6699000000953674]
KD2 pt test accuracy =  [0.474700003862381, 0.5041999816894531, 0.5203999876976013, 0.5543000102043152, 0.5726000070571899, 0.5859000086784363, 0.5940999984741211, 0.6140999794006348, 0.6184999942779541, 0.6116999983787537, 0.6240000128746033, 0.6326000094413757, 0.6348999738693237, 0.6276000142097473, 0.6403999924659729, 0.6556000113487244, 0.6431999802589417, 0.6498000025749207, 0.6438999772071838, 0.661899983882904, 0.666700005531311, 0.6579999923706055, 0.6675999760627747, 0.6705999970436096, 0.6577000021934509, 0.6690999865531921, 0.6650000214576721, 0.6723999977111816, 0.6654000282287598, 0.6743000149726868, 0.6748999953269958, 0.6686000227928162, 0.6660000085830688, 0.6764000058174133, 0.6811000108718872, 0.6791999936103821, 0.6696000099182129, 0.6764000058174133, 0.6854000091552734, 0.6823999881744385, 0.6870999932289124, 0.6775000095367432, 0.6881999969482422, 0.6780999898910522, 0.690500020980835, 0.6861000061035156, 0.6886000037193298, 0.6805999875068665, 0.6741999983787537, 0.6765000224113464, 0.678600013256073, 0.6840000152587891, 0.6818000078201294, 0.6650000214576721, 0.6665999889373779, 0.677299976348877, 0.6888999938964844, 0.6746000051498413, 0.6801000237464905, 0.6895999908447266, 0.6880999803543091, 0.6891999840736389, 0.6854000091552734, 0.6804999709129333, 0.6901999711990356, 0.6916000247001648, 0.6833999752998352, 0.6915000081062317, 0.6895999908447266, 0.6919999718666077, 0.6947000026702881, 0.6924999952316284, 0.6852999925613403, 0.6876000165939331, 0.6802999973297119, 0.6880999803543091, 0.6825000047683716, 0.6894000172615051, 0.6916000247001648, 0.6905999779701233, 0.6930999755859375, 0.6920999884605408, 0.6880999803543091, 0.6919000148773193, 0.6919000148773193, 0.6980999708175659, 0.6913999915122986, 0.692799985408783, 0.6876999735832214, 0.6953999996185303, 0.6953999996185303, 0.6942999958992004, 0.6929000020027161, 0.6843000054359436, 0.6883000135421753, 0.6610999703407288, 0.6945000290870667, 0.6915000081062317, 0.6966999769210815, 0.689300000667572, 0.6855000257492065, 0.6901999711990356, 0.6919000148773193, 0.6697999835014343, 0.6854000091552734, 0.6814000010490417, 0.6868000030517578, 0.6973999738693237, 0.6966000199317932, 0.680899977684021, 0.6922000050544739, 0.6962000131607056, 0.6848999857902527, 0.6895999908447266, 0.652999997138977, 0.6848000288009644, 0.6894000172615051, 0.7039999961853027, 0.6974999904632568, 0.699999988079071, 0.6898999810218811, 0.7045000195503235, 0.7005000114440918, 0.6919000148773193, 0.6931999921798706, 0.6690999865531921, 0.6945000290870667, 0.6947000026702881, 0.6988000273704529, 0.692300021648407, 0.6955999732017517, 0.6920999884605408, 0.689300000667572, 0.6952999830245972, 0.6769999861717224, 0.6995999813079834, 0.6902999877929688, 0.6758000254631042, 0.6941999793052673, 0.6937999725341797, 0.6851000189781189, 0.6940000057220459, 0.6814000010490417, 0.6926000118255615, 0.6916000247001648, 0.690500020980835, 0.6901000142097473, 0.6927000284194946, 0.6898999810218811]

lr = 1e-4 alpha = 0.95 beta = 0.9 temp = 10 - out.309004 pre-train start = 50 Batchnorm

lr = 1e-4 alpha = 0.95 beta = 0.9 temp = 10 - out.309171 pre-train start = 50 Dropout

"""
import sys
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy
#%%
# import importlib
# importlib.reload(misc)
# from misc import Logger
import os 
import sys

######################## Network Parameters ##################################

if len(sys.argv) == 1:
    st_parameters = {
    'lr' : 1,#float(sys.argv[1]),
    'epochs' : 5,#int(sys.argv[2]),
    "student_fst_learning" : 1,#int(sys.argv[3]), #The first learning stage of the student - number of epochs
    'alpha': 0.5,#float(sys.argv[4]), #KD weights
    'temp' : 12,#int(sys.argv[5]),   #KD weights
    'beta' : 0.9#float(sys.argv[6]), #features st weights
    
    }
else:
    st_parameters = {
    'lr' : float(sys.argv[1]),
    'epochs' : int(sys.argv[2]),
    "student_fst_learning" : int(sys.argv[3]), #The first learning stage of the student - number of epochs
    'alpha': float(sys.argv[4]), #KD weights
    'temp' : int(sys.argv[5]),   #KD weights
    'beta' : float(sys.argv[6]), #features st weights
    
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
        HR_data , y = data
        # Forward pass of teacher
        
        teacher_features, teacher_predictions = self.teacher.call(HR_data, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_features, student_predictions = self.student(HR_data, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.log_softmax(student_predictions / self.temperature, axis=1),
            )
            loss = (1 - self.alpha) * student_loss + (self.alpha * self.temperature * self.temperature) * distillation_loss

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
    
class Distiller2(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller2, self).__init__()
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
        super(Distiller2, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        HR_data , y = data
        # Forward pass of teacher
        
        teacher_features, teacher_predictions = self.teacher.call(HR_data, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_features, student_predictions = self.student(HR_data, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = (1-self.alpha) * student_loss + (self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performancekeras.layers.TimeDistributed(
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
        HR_data , y = data
        # Forward pass of teacher
        teacher_features, teacher_predictions = self.teacher(HR_data, training=False)
        # layer_name = 'teacher_features'
        # intermediate_layer_model = keras.Model(inputs=model.input,
        #                                outputs=model.get_layer(layer_name).output)
        # intermediate_output = intermediate_layer_model(data)
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_features, student_predictions = self.student(HR_data, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            features_loss = self.features_loss_fn(
                tf.nn.softmax(teacher_features / self.temperature, axis=1),
                tf.nn.softmax(student_features / self.temperature, axis=1),
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
        _, y_prediction = self.student(x, training=False)
        print(y_prediction.shape)
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
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'valid')(inputA)
    print(x1.shape)
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)

    x1=keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'valid')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)

    x1=keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'valid')(x1)
    x1=keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'valid')(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1=keras.layers.Dropout(dropout)(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)st_parameters['epochs']

    x1 = keras.layers.Flatten(name = 'teacher_features')(x1)
    x1 = keras.layers.Dense(512, activation = 'relu')(x1)
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

def student(hidden_size = 128,input_size = 32, 
            concat = True, cnn_dropout = 0.2, rnn_dropout = 0.2):
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
        DESCRIPTION. The default is 32.epochshidden_size

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    '''
    inputA = keras.layers.Input(shape=(input_size,input_size,3))
    

    # define CNN model

    x1=keras.layers.Conv2D(16,(3,3),activation='relu', padding = 'same')(inputA)
    x1=keras.layers.Dropout(cnn_dropout)(x1)
    #x1=keras.layers.BatchNormalization(momentum=0.1, epsilon = 1e-5)(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    
    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Dropout(cnn_dropout)(x1)
    #x1=keras.layers.BatchNormalization(momentum=0.1, epsilon = 1e-5)(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)

    x1=keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(x1)
    x1=keras.layers.Dropout(cnn_dropout)(x1)
    #x1=keras.layers.BatchNormalization(momentum=0.1, epsilon = 1e-5)(x1)
    x1=keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    print(x1.shape)


    x1=keras.layers.Flatten(name = 'cnn_student_features')(x1)
    print(x1.shape)

    x = keras.layers.Dense(64, activation = 'relu', name = "student_features")(x1)
    print(x1.shape)
    
    x = keras.layers.Dense(10)(x1) #activation will be in the distiller
    model = keras.models.Model(inputs=inputA,outputs=[x1,x], name = 'student{}'.format(concat))

    return model

#%%
accur_dataset = pd.DataFrame()
loss_dataset = pd.DataFrame()
#%%
teacher_network = teacher(input_size = 32, dropout = 0.2)

#%%
print('######################### TRAIN TEACHER ##############################')
teacher_history = teacher_network.fit(
                           trainX,
                           trainy,
                           batch_size = 64,
                           epochs = st_parameters['epochs'],
                           validation_data = (testX,testy),
                           verbose = 0,
                           )
print('teacher test accuracy = ',teacher_history.history['val_sparse_categorical_accuracy'])

accur_dataset['teacher_test'] = teacher_history.history['val_sparse_categorical_accuracy']
accur_dataset['teacher_train'] = teacher_history.history['sparse_categorical_accuracy']

#%%
print('######################### TRAIN STUDENT ##############################')

student_network = student(hidden_size = 128,input_size = 32, 
            concat = True, cnn_dropout = 0.2 , rnn_dropout = 0.2)


#keras.utils.plot_model(student_network, expand_nested=True)
#%%
print('################## Train Student - No pre-training ####################')
print('####################### Knowledge Distillation ########################')
KD_student = keras.models.clone_model(student_network)
KD_student.set_weights(student_network.get_weights()) 
KD = Distiller(KD_student, teacher_network)

KD.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           distillation_loss_fn = keras.losses.KLDivergence(),
           alpha = st_parameters['alpha'],
           temperature = st_parameters['temp'])

KD_history = KD.fit(trainX,
        trainy,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (testX,testy),
        verbose = 0,
        )

print('KD test accuracy = ',KD_history.history['val_sparse_categorical_accuracy'])
print('KD distillation loss = ',KD_history.history['distillation_loss'])
loss_dataset['KD'] = KD_history.history['distillation_loss']
accur_dataset['KD_test'] = KD_history.history['val_sparse_categorical_accuracy']
accur_dataset['KD_train'] = KD_history.history['sparse_categorical_accuracy']

#%%
print('################## Train Student - No pre-training ####################')
print('####################### Knowledge Distillation 2 ########################')
KD_student2 = keras.models.clone_model(student_network)
KD_student2.set_weights(student_network.get_weights()) 
KD2 = Distiller2(KD_student2, teacher_network)

KD2.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           distillation_loss_fn = keras.losses.KLDivergence(),
           alpha = st_parameters['alpha'],
           temperature = st_parameters['temp'])

KD2_history = KD2.fit(trainX,
        trainy,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (testX,testy),
        verbose = 0,
        )

print('KD2 test accuracy = ',KD2_history.history['val_sparse_categorical_accuracy'])
print('KD2 distillation loss = ',KD2_history.history['distillation_loss'])
loss_dataset['KD2'] = KD2_history.history['distillation_loss']
accur_dataset['KD2_test'] = KD2_history.history['val_sparse_categorical_accuracy']
accur_dataset['KD2_train'] = KD2_history.history['sparse_categorical_accuracy']
#%%
print('########################## Feature Learing ############################')
FL_student = keras.models.clone_model(student_network)
FL_student.set_weights(student_network.get_weights()) 
FL = feature_st(FL_student, teacher_network)

FL.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           features_loss_fn = keras.losses.MeanSquaredError(),
           beta = st_parameters['beta'],
           temperature = st_parameters['temp'])

FL_history = FL.fit(trainX,
        trainy,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (testX,testy),
        verbose = 0,
        )

print('FL test accuracy = ',FL_history.history['val_sparse_categorical_accuracy'])
print('FL feature loss = ', FL_history.history['features_loss'])
accur_dataset['FL_test'] = FL_history.history['val_sparse_categorical_accuracy']
accur_dataset['FL_train'] = FL_history.history['sparse_categorical_accuracy']
#%%
print('############################## Baseline ###############################')
baseline_student = keras.models.clone_model(student_network)
baseline_student.set_weights(student_network.get_weights()) 
model = teacher_training(baseline_student)
model.compile(
        optimizer=keras.optimizers.Adam(lr = st_parameters['lr']),
        loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
baseline_history = model.fit(trainX,
        trainy,
        batch_size = 64,
        epochs = st_parameters['epochs'],
        validation_data = (testX,testy),
        verbose = 0,
        )

print('Baseline test accuracy = ',baseline_history.history['val_sparse_categorical_accuracy'])
accur_dataset['base_test'] = baseline_history.history['val_sparse_categorical_accuracy']
accur_dataset['base_train'] = baseline_history.history['sparse_categorical_accuracy']

#%%
plt.figure()
#plt.plot(teacher_history.history['val_sparse_categorical_accuracy'], label = 'teacher')
plt.plot(KD_history.history['val_sparse_categorical_accuracy'], label = 'KD')
plt.plot(KD2_history.history['val_sparse_categorical_accuracy'], label = 'KD2')
plt.plot(FL_history.history['val_sparse_categorical_accuracy'], label = 'FL')
plt.plot(baseline_history.history['val_sparse_categorical_accuracy'], label = 'baseline')
plt.legend()
plt.title('Comparing KD and FL teacher student models on cifar')
plt.savefig('KD_FL_cifar_syclop_{:.0e}_{}.png'.format(st_parameters['alpha'],st_parameters['temp']))

#%%
print('################ Train Student - with pre-training ####################')
base_student = keras.models.clone_model(student_network)
base_student.set_weights(student_network.get_weights()) 
base_model = teacher_training(base_student)
base_model.compile(
        optimizer=keras.optimizers.Adam(lr = st_parameters['lr']),
        loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
base_history = base_model.fit(trainX,
        trainy,
        batch_size = 64,
        epochs = st_parameters['student_fst_learning'],
        validation_data = (testX,testy),
        verbose = 0,
        )

print('Base test accuracy = ',base_history.history['val_sparse_categorical_accuracy'])

#%%
new_epochs = st_parameters['epochs'] - st_parameters['student_fst_learning']
if new_epochs < 1:
    new_epochs = 1
    print("ERROR - less than 1 epochs left after base learning!!!!!!!!")
print('############## Knowledge Distillation wt Pre-Trainiong ################')
KD_student_pt = keras.models.clone_model(base_student)
KD_student_pt.set_weights(base_student.get_weights()) 
KD_pt = Distiller(KD_student_pt, teacher_network)

KD_pt.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           distillation_loss_fn = keras.losses.KLDivergence(),
           alpha = st_parameters['alpha'],
           temperature = st_parameters['temp'])

#KD_pt.evaluate(testX,testy)
KD_pt_history = KD_pt.fit(trainX,
        trainy,
        batch_size = 64,
        epochs = new_epochs,
        validation_data = (testX,testy),
        verbose = 0,
        )

print('KD pt test accuracy = ',KD_pt_history.history['val_sparse_categorical_accuracy'])
print('KD pt distillation loss = ',KD_pt_history.history['distillation_loss'])
KD_pt_dist_loss = np.ones(len(KD_history.history['distillation_loss']))*KD_pt_history.history['distillation_loss'][-1]
KD_pt_dist_loss[:len(KD_pt_history.history['distillation_loss'])] = KD_pt_history.history['distillation_loss']
loss_dataset['KD_pt'] = KD_pt_dist_loss
accur_dataset['KD_pt_test'] = base_history.history['val_sparse_categorical_accuracy'] + \
                                KD_pt_history.history['val_sparse_categorical_accuracy']
accur_dataset['KD_pt_train'] = base_history.history['sparse_categorical_accuracy'] + \
                                KD_pt_history.history['sparse_categorical_accuracy']
#%%

print('############## Knowledge Distillation 2 wt Pre-Trainiong ################')
KD2_student_pt = keras.models.clone_model(base_student)
KD2_student_pt.set_weights(base_student.get_weights()) 
KD2_pt = Distiller2(KD2_student_pt, teacher_network)

KD2_pt.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           distillation_loss_fn = keras.losses.KLDivergence(),
           alpha = st_parameters['alpha'],
           temperature = st_parameters['temp'])

#KD2_pt.evaluate(testX,testy)
KD2_pt_history = KD2_pt.fit(trainX,
        trainy,
        batch_size = 64,
        epochs = new_epochs,
        validation_data = (testX,testy),
        verbose = 0,
        )

print('KD2 pt test accuracy = ',KD2_pt_history.history['val_sparse_categorical_accuracy'])
print('KD2 pt distillation loss = ',KD2_pt_history.history['distillation_loss'])
KD2_pt_dist_loss = np.ones(len(KD2_history.history['distillation_loss']))*KD_pt_history.history['distillation_loss'][-1]
KD2_pt_dist_loss[:len(KD2_pt_history.history['distillation_loss'])] = KD2_pt_history.history['distillation_loss']
loss_dataset['KD2_pt'] = KD2_pt_dist_loss
accur_dataset['KD2_pt_test'] = base_history.history['val_sparse_categorical_accuracy'] + \
                                KD2_pt_history.history['val_sparse_categorical_accuracy']
accur_dataset['KD2_pt_train'] = base_history.history['sparse_categorical_accuracy'] + \
                                KD2_pt_history.history['sparse_categorical_accuracy']
#%%
print('########################## Feature Learing ############################')
FL_pt_student = keras.models.clone_model(base_student)
FL_pt_student.set_weights(base_student.get_weights()) 
FL_pt = feature_st(FL_pt_student, teacher_network)

FL_pt.compile(optimizer = keras.optimizers.Adam(lr = st_parameters['lr']),
           metrics   = ["sparse_categorical_accuracy"],
           student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           features_loss_fn = keras.losses.MeanSquaredError(),
           beta = st_parameters['beta'],
           temperature = st_parameters['temp'])
#FL_pt.evaluate(testX,testy)
FL_pt_history = FL_pt.fit(trainX,
        trainy,
        batch_size = 64,
        epochs = new_epochs,
        validation_data = (testX,testy),
        verbose = 0,
        )


print('FL pt test accuracy = ',FL_pt_history.history['val_sparse_categorical_accuracy'])
print('FL pt feature loss = ', FL_pt_history.history['features_loss'])
accur_dataset['FL_pt_test'] = base_history.history['val_sparse_categorical_accuracy'] + \
                                FL_pt_history.history['val_sparse_categorical_accuracy']
accur_dataset['FL_pt_train'] = base_history.history['val_sparse_categorical_accuracy'] + \
                                FL_pt_history.history['sparse_categorical_accuracy']

accur_dataset.to_pickle('HR_KD_FL_pt_cifar_syclop_{:.0e}_{}_{:.0e}.pkl'.format(st_parameters['alpha'],st_parameters['temp'], st_parameters['beta']))
#%%
plt.figure()
#plt.plot(teacher_history.history['val_sparse_categorical_accuracy'], label = 'teacher')
plt.plot(KD_history.history['val_sparse_categorical_accuracy'], label = 'KD')
plt.plot(KD2_history.history['val_sparse_categorical_accuracy'], label = 'KD2')
plt.plot(FL_history.history['val_sparse_categorical_accuracy'], label = 'FL')
plt.plot(baseline_history.history['val_sparse_categorical_accuracy'], label = 'baseline')
plt.plot(accur_dataset['KD_pt_test'], label = 'KD_pt')
plt.plot(accur_dataset['KD2_pt_test'], label = 'KD2_pt')
plt.plot(accur_dataset['FL_pt_test'], label = 'FL_pt')
plt.legend()
plt.title('Comparing KD and FL teacher student models on cifar')
plt.savefig('HR_KD_FL_pt_cifar_syclop_{:.0e}_{}_{:.0e}.png'.format(st_parameters['alpha'],st_parameters['temp'], st_parameters['beta']))





