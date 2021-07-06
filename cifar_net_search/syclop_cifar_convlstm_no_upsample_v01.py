'''
The follwing code runs a test lstm network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 

One convlstm layer right after the first cnn 

cnn_dropout = 0.4 rnn_dropout = 0.2 , WITH cnivlstm_dropout samples = 10, h = 256, epochs = 50, convlstm activation = relu - out.373233 (Based on best results from cnn - gru)
################# convlstm_cnn_mix_v0_True Validation Accuracy =  [0.3555999994277954, 0.37619999051094055, 0.4235999882221222, 0.4275999963283539, 0.46779999136924744, 0.4819999933242798, 0.48980000615119934, 0.4968000054359436, 0.5194000005722046, 0.5131999850273132, 0.5242000222206116, 0.5284000039100647, 0.5144000053405762, 0.5365999937057495, 0.5465999841690063, 0.5496000051498413, 0.5407999753952026, 0.5626000165939331, 0.5573999881744385, 0.5558000206947327, 0.5586000084877014, 0.5645999908447266, 0.5717999935150146, 0.569599986076355, 0.5727999806404114, 0.5630000233650208, 0.571399986743927, 0.5580000281333923, 0.5852000117301941, 0.5753999948501587, 0.5738000273704529, 0.579200029373169, 0.5726000070571899, 0.5789999961853027, 0.5648000240325928, 0.5776000022888184, 0.5763999819755554, 0.5799999833106995, 0.58160001039505, 0.5839999914169312, 0.5860000252723694, 0.5834000110626221, 0.5825999975204468, 0.5860000252723694, 0.5807999968528748, 0.5831999778747559, 0.5789999961853027, 0.5758000016212463, 0.58160001039505, 0.5807999968528748]
################# convlstm_cnn_mix_v0_True Training Accuracy =  [0.25715556740760803, 0.36764445900917053, 0.4077777862548828, 0.4264444410800934, 0.4399999976158142, 0.4568444490432739, 0.46862220764160156, 0.4764222204685211, 0.4854666590690613, 0.4959777891635895, 0.5049999952316284, 0.5096889138221741, 0.5180666446685791, 0.5244666934013367, 0.5313777923583984, 0.536133348941803, 0.5363110899925232, 0.5413333177566528, 0.549311101436615, 0.5522222518920898, 0.557022213935852, 0.5595999956130981, 0.563955545425415, 0.5663999915122986, 0.5641111135482788, 0.5719777941703796, 0.5754444599151611, 0.5798222422599792, 0.580644428730011, 0.5836222171783447, 0.5849555730819702, 0.586222231388092, 0.5915555357933044, 0.594355583190918, 0.5944888591766357, 0.5996444225311279, 0.6037111282348633, 0.6028888821601868, 0.6078444719314575, 0.6094889044761658, 0.6093555688858032, 0.6102889180183411, 0.6134666800498962, 0.6152889132499695, 0.6180889010429382, 0.6205999851226807, 0.6215111017227173, 0.6247333288192749, 0.6240666508674622, 0.6295999884605408]

cnn_dropout = 0.4 rnn_dropout = 0.2 , WITH cnivlstm_dropout samples = 10, h = 256, epochs = 150, convlstm activation = relu - out.373440 (Based on best results from cnn - gru)
################# convlstm_cnn_mix_v01_True Validation Accuracy =  [0.31619998812675476, 0.4059999883174896, 0.4047999978065491, 0.436599999666214, 0.45899999141693115, 0.4731999933719635, 0.4819999933242798, 0.487199991941452, 0.4936000108718872, 0.4991999864578247, 0.5113999843597412, 0.5221999883651733, 0.5235999822616577, 0.5278000235557556, 0.5368000268936157, 0.5289999842643738, 0.5253999829292297, 0.5428000092506409, 0.5382000207901001, 0.5414000153541565, 0.5450000166893005, 0.5511999726295471, 0.5523999929428101, 0.5672000050544739, 0.5468000173568726, 0.5619999766349792, 0.5522000193595886, 0.5722000002861023, 0.5673999786376953, 0.5720000267028809, 0.5690000057220459, 0.5726000070571899, 0.5763999819755554, 0.571399986743927, 0.5673999786376953, 0.5722000002861023, 0.5691999793052673, 0.5781999826431274, 0.578000009059906, 0.5795999765396118, 0.5748000144958496, 0.5888000130653381, 0.5809999704360962, 0.5821999907493591, 0.5759999752044678, 0.5771999955177307, 0.5705999732017517, 0.5825999975204468, 0.5770000219345093, 0.5738000273704529, 0.5756000280380249, 0.5752000212669373, 0.5802000164985657, 0.5781999826431274, 0.5709999799728394, 0.5717999935150146, 0.5781999826431274, 0.5831999778747559, 0.5812000036239624, 0.5849999785423279, 0.5789999961853027, 0.5756000280380249, 0.5758000016212463, 0.5771999955177307, 0.5856000185012817, 0.5785999894142151, 0.5720000267028809, 0.5734000205993652, 0.5741999745368958, 0.5824000239372253, 0.5776000022888184, 0.5756000280380249, 0.5687999725341797, 0.5676000118255615, 0.5738000273704529, 0.5857999920845032, 0.5799999833106995, 0.5766000151634216, 0.5788000226020813, 0.5831999778747559, 0.5753999948501587, 0.5734000205993652, 0.5631999969482422, 0.5687999725341797, 0.5788000226020813, 0.578000009059906, 0.5756000280380249, 0.5708000063896179, 0.5618000030517578, 0.5708000063896179, 0.5741999745368958, 0.5753999948501587, 0.5720000267028809, 0.5722000002861023, 0.5637999773025513, 0.574999988079071, 0.5690000057220459, 0.58160001039505, 0.5694000124931335, 0.5676000118255615, 0.5738000273704529, 0.5651999711990356, 0.574999988079071, 0.5703999996185303, 0.569599986076355, 0.5690000057220459, 0.5672000050544739, 0.5655999779701233, 0.5637999773025513, 0.5691999793052673, 0.5702000260353088, 0.5680000185966492, 0.5613999962806702, 0.5662000179290771, 0.5623999834060669, 0.567799985408783, 0.5662000179290771, 0.5712000131607056, 0.5673999786376953, 0.5630000233650208, 0.5594000220298767, 0.5608000159263611, 0.5681999921798706, 0.5659999847412109, 0.5630000233650208, 0.5619999766349792, 0.5626000165939331, 0.5654000043869019, 0.5654000043869019, 0.5631999969482422, 0.5680000185966492, 0.5601999759674072, 0.5586000084877014, 0.5590000152587891, 0.5605999827384949, 0.5558000206947327, 0.5636000037193298, 0.555400013923645, 0.5644000172615051, 0.5595999956130981, 0.5608000159263611, 0.5685999989509583, 0.5626000165939331, 0.5590000152587891, 0.5623999834060669, 0.5541999936103821, 0.5523999929428101, 0.5519999861717224, 0.5582000017166138, 0.5558000206947327]
################# convlstm_cnn_mix_v01_True Training Accuracy =  [0.249466672539711, 0.358822226524353, 0.39959999918937683, 0.41804444789886475, 0.43524444103240967, 0.4512222111225128, 0.4600444436073303, 0.46995556354522705, 0.47760000824928284, 0.48697778582572937, 0.4950222074985504, 0.49779999256134033, 0.5047777891159058, 0.5094444155693054, 0.5156221985816956, 0.5198000073432922, 0.5297999978065491, 0.5285778045654297, 0.5348444581031799, 0.5395777821540833, 0.5428000092506409, 0.5471110939979553, 0.5546444654464722, 0.5557777881622314, 0.5594000220298767, 0.5629777908325195, 0.5652666687965393, 0.5679110884666443, 0.5738222002983093, 0.5741999745368958, 0.5807777643203735, 0.5786888599395752, 0.5838666558265686, 0.5865111351013184, 0.5898444652557373, 0.592199981212616, 0.5955111384391785, 0.5958889126777649, 0.6001777648925781, 0.6027555465698242, 0.6022666692733765, 0.606844425201416, 0.6050666570663452, 0.6121333241462708, 0.6107555627822876, 0.6132222414016724, 0.6163111329078674, 0.6169777512550354, 0.6183333396911621, 0.622511088848114, 0.624822199344635, 0.6251555681228638, 0.6260666847229004, 0.6275110840797424, 0.6320444345474243, 0.6323999762535095, 0.6347333192825317, 0.6363333463668823, 0.6373777985572815, 0.6391111016273499, 0.6382444500923157, 0.63919997215271, 0.6450222134590149, 0.6434000134468079, 0.6443555355072021, 0.6475555300712585, 0.6497777700424194, 0.6504666805267334, 0.6514222025871277, 0.6546000242233276, 0.6522889137268066, 0.6551111340522766, 0.6570000052452087, 0.6564444303512573, 0.658466637134552, 0.6592222452163696, 0.662066638469696, 0.6632444262504578, 0.6638222336769104, 0.6681333184242249, 0.6650221943855286, 0.6649555563926697, 0.6700000166893005, 0.6726666688919067, 0.6718888878822327, 0.6709111332893372, 0.6744444370269775, 0.6745333075523376, 0.6758221983909607, 0.6766666769981384, 0.6747111082077026, 0.6774666905403137, 0.6812666654586792, 0.6800888776779175, 0.6768222451210022, 0.6801333427429199, 0.6830888986587524, 0.6844000220298767, 0.6855999827384949, 0.6861777901649475, 0.686822235584259, 0.6895111203193665, 0.6921555399894714, 0.6860666871070862, 0.6852666735649109, 0.6915333271026611, 0.6921333074569702, 0.6931777596473694, 0.6941555738449097, 0.6948666572570801, 0.6968888640403748, 0.6943777799606323, 0.6979555487632751, 0.6966888904571533, 0.6968888640403748, 0.6993555426597595, 0.6975555419921875, 0.7030444741249084, 0.6989777684211731, 0.7029555439949036, 0.7020221948623657, 0.7024222016334534, 0.7059333324432373, 0.7076444625854492, 0.7047333121299744, 0.7077111005783081, 0.7068444490432739, 0.7082666754722595, 0.7079333066940308, 0.7075555324554443, 0.7076888680458069, 0.7130222320556641, 0.71224445104599, 0.7111555337905884, 0.7123333215713501, 0.7143333554267883, 0.7105333209037781, 0.718155562877655, 0.7120888829231262, 0.7151333093643188, 0.7185778021812439, 0.7164000272750854, 0.7191110849380493, 0.7173110842704773, 0.7188666462898254, 0.7190889120101929, 0.7207777500152588, 0.7199777960777283, 0.7187111377716064, 0.722000002861023]

################# convlstm_cnn_mix_v0_True Validation Accuracy =  [0.3555999994277954, 0.37619999051094055, 0.4235999882221222, 0.4275999963283539, 0.46779999136924744, 0.4819999933242798, 0.48980000615119934, 0.4968000054359436, 0.5194000005722046, 0.5131999850273132, 0.5242000222206116, 0.5284000039100647, 0.5144000053405762, 0.5365999937057495, 0.5465999841690063, 0.5496000051498413, 0.5407999753952026, 0.5626000165939331, 0.5573999881744385, 0.5558000206947327, 0.5586000084877014, 0.5645999908447266, 0.5717999935150146, 0.569599986076355, 0.5727999806404114, 0.5630000233650208, 0.571399986743927, 0.5580000281333923, 0.5852000117301941, 0.5753999948501587, 0.5738000273704529, 0.579200029373169, 0.5726000070571899, 0.5789999961853027, 0.5648000240325928, 0.5776000022888184, 0.5763999819755554, 0.5799999833106995, 0.58160001039505, 0.5839999914169312, 0.5860000252723694, 0.5834000110626221, 0.5825999975204468, 0.5860000252723694, 0.5807999968528748, 0.5831999778747559, 0.5789999961853027, 0.5758000016212463, 0.58160001039505, 0.5807999968528748]
################# convlstm_cnn_mix_v0_True Training Accuracy =  [0.25715556740760803, 0.36764445900917053, 0.4077777862548828, 0.4264444410800934, 0.4399999976158142, 0.4568444490432739, 0.46862220764160156, 0.4764222204685211, 0.4854666590690613, 0.4959777891635895, 0.5049999952316284, 0.5096889138221741, 0.5180666446685791, 0.5244666934013367, 0.5313777923583984, 0.536133348941803, 0.5363110899925232, 0.5413333177566528, 0.549311101436615, 0.5522222518920898, 0.557022213935852, 0.5595999956130981, 0.563955545425415, 0.5663999915122986, 0.5641111135482788, 0.5719777941703796, 0.5754444599151611, 0.5798222422599792, 0.580644428730011, 0.5836222171783447, 0.5849555730819702, 0.586222231388092, 0.5915555357933044, 0.594355583190918, 0.5944888591766357, 0.5996444225311279, 0.6037111282348633, 0.6028888821601868, 0.6078444719314575, 0.6094889044761658, 0.6093555688858032, 0.6102889180183411, 0.6134666800498962, 0.6152889132499695, 0.6180889010429382, 0.6205999851226807, 0.6215111017227173, 0.6247333288192749, 0.6240666508674622, 0.6295999884605408]

with 20 samples and epochs = 50, h=128, out.980236

with 20 samples and epochs = 50, h=256, out.980276
'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy


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

kernel_regularizer_list = [None, keras.regularizers.l1(),keras.regularizers.l2(),keras.regularizers.l1_l2()]
optimizer_list = [tf.keras.optimizers.Adam, tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop]
if len(sys.argv) > 1:
    paramaters = {
    'epochs' : int(sys.argv[1]),
    
    'sample' : int(sys.argv[2]),
    
    'res' : int(sys.argv[3]),
    
    'hidden_size' : int(sys.argv[4]),
    
    'concat' : int(sys.argv[5]),
    
    'regularizer' : keras.regularizers.l1(),#kernel_regularizer_list[int(sys.argv[6])],
    
    'optimizer' : optimizer_list[int(sys.argv[7])],
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4,
    
    'run_id' : np.random.randint(1000,9000)
    }
    
else:
    paramaters = {
    'epochs' : 1,
    
    'sample' : 5,
    
    'res' : 8,
    
    'hidden_size' : 128,
    
    'concat' : 1,
    
    'regularizer' : None,
    
    'optimizer' : optimizer_list[0],
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4,
    
    'run_id' : np.random.randint(1000,9000)
    }
   
print(paramaters)
for key,val in paramaters.items():
    exec(key + '=val')
epochs = epochs
sample = sample 
res = res 
hidden_size =hidden_size
concat = concat
regularizer = regularizer
optimizer = optimizer
cnn_dropout = cnn_dropout
rnn_dropout = rnn_dropout
lr = lr
run_id = run_id
n_timesteps = sample


def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1),np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

def convlstm(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True):
    '''
    
    CNN RNN combination that extends the CNN to a network that achieves 
    ~80% accuracy on full res cifar.

    Parameters
    ----------
    n_timesteps : TYPE, optional
        DESCRIPTION. The default is 5.
    img_dim : TYPE, optional
        DESCRIPTION. The default is 32.
    hidden_size : TYPE, optional
        DESCRIPTION. The default is 128.
    input_size : TYPE, optional
        DESCRIPTION. The default is 32.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    '''
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,3))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3), activation='relu',padding = 'same'))(inputA)
    x1=keras.layers.ConvLSTM2D(32,(3,3), padding = 'same', dropout = cnn_dropout, recurrent_dropout=rnn_dropout,return_sequences=True)(x1)
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


    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    print(x1.shape)
    if concat:
        x = keras.layers.Concatenate()([x1,inputB])
    else:
        x = x1
    print(x.shape)

    # define LSTM model
    x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),return_sequences=True,recurrent_dropout=rnn_dropout)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'convlstm_cnn_mix_v01_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = convlstm(n_timesteps = sample, hidden_size = hidden_size,input_size = res, concat = True)
#keras.utils.plot_model(rnn_net, expand_nested=True,  to_file='{}.png'.format(rnn_net.name))
#cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = res, dropout = cnn_dropout)


# hp = HP()
# hp.save_path = 'saved_runs'

# hp.description = "syclop cifar net search runs"
# hp.this_run_name = 'syclop_{}'.format(rnn_net.name)
# deploy_logs()

train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )
                                    #bad_res_func = bad_res101, up_sample = True)

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)


print("##################### Fit {} and trajectories model on training data res = {} ##################".format(rnn_net.name,res))
rnn_history = rnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 0)

# print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])
# print('################# {} Training Accuracy = '.format(cnn_net.name),rnn_history.history['sparse_categorical_accuracy'])

print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
# plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
# plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
plt.legend()
plt.title('{} on cifar res = {} hs = {} dropout = {}, num samples = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout,sample))
plt.savefig('{} on Cifar res = {}, no upsample, val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict{}_{}'.format(rnn_net.name, hidden_size,cnn_dropout), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
# with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict'.format(cnn_net.name), 'wb') as file_pi:
#     pickle.dump(cnn_history.history, file_pi)
dataset_update(rnn_history, rnn_net,paramaters)    
write_to_file(rnn_history, rnn_net,paramaters)    
    