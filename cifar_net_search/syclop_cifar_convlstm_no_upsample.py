'''
The follwing code runs a test lstm network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 

cnn_dropout = 0.4 rnn_dropout = 0.2, no convlstm_dropout samples = 10, h = 256, epochs = 150 - convlstm activation = relu out.276576 (Based on best results from cnn - gru)
################ convlstm_cnn_mix_v1_True Validation Accuracy =  [0.3061999976634979, 0.40459999442100525, 0.4521999955177307, 0.47040000557899475, 0.4740000069141388, 0.5077999830245972, 0.4867999851703644, 0.5116000175476074, 0.5383999943733215, 0.5342000126838684, 0.5483999848365784, 0.5491999983787537, 0.5447999835014343, 0.548799991607666, 0.5562000274658203, 0.553600013256073, 0.550599992275238, 0.5296000242233276, 0.5450000166893005, 0.571399986743927, 0.5551999807357788, 0.5504000186920166, 0.5523999929428101, 0.5532000064849854, 0.5450000166893005, 0.5558000206947327, 0.5644000172615051, 0.5532000064849854, 0.5475999712944031, 0.5508000254631042, 0.5511999726295471, 0.5546000003814697, 0.5540000200271606, 0.5475999712944031, 0.5483999848365784, 0.5478000044822693, 0.5573999881744385, 0.5551999807357788, 0.5386000275611877, 0.5374000072479248, 0.5455999970436096, 0.5465999841690063, 0.5370000004768372, 0.5442000031471252, 0.534600019454956, 0.5396000146865845, 0.5388000011444092, 0.519599974155426, 0.5411999821662903, 0.5364000201225281, 0.5181999802589417, 0.5288000106811523, 0.5285999774932861, 0.5248000025749207, 0.5378000140190125, 0.5333999991416931, 0.5275999903678894, 0.5375999808311462, 0.5353999733924866, 0.524399995803833, 0.5324000120162964, 0.5296000242233276, 0.5220000147819519, 0.5249999761581421, 0.5307999849319458, 0.515999972820282, 0.5353999733924866, 0.525600016117096, 0.5297999978065491, 0.5188000202178955, 0.5260000228881836, 0.5199999809265137, 0.517799973487854, 0.5134000182151794, 0.5103999972343445, 0.5153999924659729, 0.5281999707221985, 0.5235999822616577, 0.520799994468689, 0.5216000080108643, 0.5242000222206116, 0.5162000060081482, 0.527999997138977, 0.5175999999046326, 0.5221999883651733, 0.5249999761581421, 0.5194000005722046, 0.5135999917984009, 0.5202000141143799, 0.5239999890327454, 0.5203999876976013, 0.5257999897003174, 0.5094000101089478, 0.5224000215530396, 0.519599974155426, 0.525600016117096, 0.5163999795913696, 0.5127999782562256, 0.5108000040054321, 0.5149999856948853, 0.5156000256538391, 0.5163999795913696, 0.5103999972343445, 0.5117999911308289, 0.515999972820282, 0.5174000263214111, 0.5278000235557556, 0.5094000101089478, 0.5175999999046326, 0.520799994468689, 0.5009999871253967, 0.5149999856948853, 0.515999972820282, 0.5171999931335449, 0.5149999856948853, 0.5062000155448914, 0.5049999952316284, 0.5144000053405762, 0.5212000012397766, 0.51419997215271, 0.5180000066757202, 0.5166000127792358, 0.5135999917984009, 0.5098000168800354, 0.5131999850273132, 0.5120000243186951, 0.5123999714851379, 0.5109999775886536, 0.5113999843597412, 0.5099999904632568, 0.5174000263214111, 0.5073999762535095, 0.5185999870300293, 0.5126000046730042, 0.5098000168800354, 0.5131999850273132, 0.5123999714851379, 0.5138000249862671, 0.5188000202178955, 0.5108000040054321, 0.5293999910354614, 0.525600016117096, 0.5171999931335449, 0.5077999830245972, 0.5123999714851379, 0.5099999904632568, 0.5113999843597412, 0.524399995803833, 0.5194000005722046, 0.51419997215271]
################ convlstm_cnn_mix_v1_True Training Accuracy =  [0.22942222654819489, 0.3643777668476105, 0.41440001130104065, 0.44457778334617615, 0.4670666754245758, 0.4886888861656189, 0.5065555572509766, 0.5184000134468079, 0.5296888947486877, 0.5417333245277405, 0.5516444444656372, 0.5579555630683899, 0.5675333142280579, 0.5736666917800903, 0.5827333331108093, 0.5899555683135986, 0.6020222306251526, 0.6079778075218201, 0.6134889125823975, 0.6179555654525757, 0.6273777484893799, 0.6341999769210815, 0.6447333097457886, 0.6485999822616577, 0.6494888663291931, 0.6548222303390503, 0.6620888710021973, 0.6679111123085022, 0.6749110817909241, 0.6765333414077759, 0.6862444281578064, 0.6877555847167969, 0.6923555731773376, 0.7004666924476624, 0.7001333236694336, 0.7070888876914978, 0.7110221982002258, 0.7136666774749756, 0.7207333445549011, 0.7224444150924683, 0.7276222109794617, 0.7346888780593872, 0.7361111044883728, 0.7378888726234436, 0.7410444617271423, 0.7450888752937317, 0.7458222508430481, 0.750333309173584, 0.7541333436965942, 0.7538889050483704, 0.7587777972221375, 0.7619110941886902, 0.7661555409431458, 0.768488883972168, 0.7723333239555359, 0.773711085319519, 0.7766000032424927, 0.7810888886451721, 0.7809110879898071, 0.7823110818862915, 0.78493332862854, 0.7901111245155334, 0.7889111042022705, 0.7912222146987915, 0.7937555313110352, 0.7960444688796997, 0.8001555800437927, 0.8045111298561096, 0.8038889169692993, 0.807711124420166, 0.8117333054542542, 0.8104000091552734, 0.8126888871192932, 0.8137778043746948, 0.813177764415741, 0.8199555277824402, 0.8218666911125183, 0.8213111162185669, 0.8250222206115723, 0.8240000009536743, 0.8286444544792175, 0.829800009727478, 0.8309999704360962, 0.8318444490432739, 0.8328444361686707, 0.8352888822555542, 0.8356888890266418, 0.8400222063064575, 0.8386666774749756, 0.8427333235740662, 0.8437777757644653, 0.8446000218391418, 0.8460888862609863, 0.8461333513259888, 0.8481555581092834, 0.8525111079216003, 0.848800003528595, 0.8526889085769653, 0.8531110882759094, 0.8532000184059143, 0.8565999865531921, 0.8604666590690613, 0.8532000184059143, 0.8614444732666016, 0.8590222001075745, 0.8608888983726501, 0.8641111254692078, 0.8655555844306946, 0.8672000169754028, 0.8659555315971375, 0.8683111071586609, 0.8712000250816345, 0.8698444366455078, 0.8686888813972473, 0.8723333477973938, 0.8747110962867737, 0.8724444508552551, 0.8758000135421753, 0.874666690826416, 0.877133309841156, 0.8780444264411926, 0.87782222032547, 0.8724222183227539, 0.8794222474098206, 0.8778889179229736, 0.8823999762535095, 0.8823999762535095, 0.8830666542053223, 0.8864222168922424, 0.8843777775764465, 0.8885111212730408, 0.886888861656189, 0.8881999850273132, 0.8883110880851746, 0.88637775182724, 0.8884666562080383, 0.8869110941886902, 0.8884888887405396, 0.8867777585983276, 0.8870888948440552, 0.8916000127792358, 0.8945778012275696, 0.8943111300468445, 0.895799994468689, 0.8898444175720215, 0.8869555592536926, 0.8943555355072021, 0.8945555686950684, 0.8971999883651733, 0.8980000019073486]

cnn_dropout = 0.4 rnn_dropout = 0.2, with convlstm_dropout samples = 10, h = 256, epochs = 150 - convlstm activation = relu out.297278 (Based on best results from cnn - gru)

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

# import importlib
# importlib.reload(misc)
# from misc import Logger
# import os 


# def deploy_logs():
#     if not os.path.exists(hp.save_path):
#         os.makedirs(hp.save_path)

#     dir_success = False
#     for sfx in range(1):  # todo legacy
#         candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
#         if not os.path.exists(candidate_path):
#             hp.this_run_path = candidate_path
#             os.makedirs(hp.this_run_path)
#             dir_success = Truecnn_net = cnn_one_img(n_timesteps = sample, input_size = 28, input_dim = 1)
#             break
#     if not dir_success:
#         error('run name already exists!')

#     sys.stdout = Logger(hp.this_run_path+'log.log')
#     print('results are in:', hp.this_run_path)
#     print('description: ', hp.description)
#     #print('hyper-parameters (partial):', hp.dict)
if len(sys.argv) > 1:
    paramaters = {
    'epochs' : int(sys.argv[1]),
    
    'sample' : int(sys.argv[2]),
    
    'res' : int(sys.argv[3]),
    
    'hidden_size' : int(sys.argv[4]),
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4
    }
    
else:
    paramaters = {
    'epochs' : 1,
    
    'sample' : 5,
    
    'res' : 8,
    
    'hidden_size' : 128,
    
    'cnn_dropout' : 0.4,

    'rnn_dropout' : 0.2,

    'lr' : 5e-4
    }
    
print(paramaters)
for key,val in paramaters.items():
    exec(key + '=val')
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
    x1=keras.layers.ConvLSTM2D(32,(3,3),padding = 'same', dropout = cnn_dropout, recurrent_dropout=rnn_dropout,return_sequences=True)(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.ConvLSTM2D(64,(3,3), padding = 'same', dropout = cnn_dropout, recurrent_dropout=rnn_dropout,return_sequences=True)(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(cnn_dropout))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same'))(x1)
    x1=keras.layers.ConvLSTM2D(128,(3,3),padding = 'same',dropout = cnn_dropout, recurrent_dropout=rnn_dropout,return_sequences=True)(x1)
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
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'convlstm_cnn_mix_v1_{}'.format(concat))
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
    