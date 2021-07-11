'''
The follwing code runs a test RNN with a dense layer to integrate the coordinates
network on the CIFAR dataset 

I will explicitly write the networks here for ease of understanding 

Running a net [32,64,128] only ConvLSTM without dropout
################# ConvLSTM_False Validation Accuracy =  [0.4234, 0.4724, 0.4916, 0.5356, 0.5498, 0.548, 0.5452, 0.5408, 0.5452, 0.5444, 0.536, 0.5438, 0.5256, 0.533, 0.545, 0.5322, 0.5288, 0.5406, 0.5374, 0.5382, 0.5262, 0.5358, 0.5286, 0.529, 0.539, 0.5346, 0.5432, 0.5398, 0.5298, 0.5296]
################# ConvLSTM_False Training Accuracy =  [0.352, 0.4515111, 0.51217777, 0.5662, 0.63375556, 0.7343111, 0.8472, 0.92388886, 0.95086664, 0.9685111, 0.9725111, 0.97726667, 0.97882223, 0.9786889, 0.98053336, 0.98477775, 0.98322225, 0.9847556, 0.98584443, 0.9860889, 0.9863333, 0.9879778, 0.98895556, 0.9884, 0.9898889, 0.98928887, 0.9894222, 0.99126667, 0.99175555, 0.9908]
Running a net [32,64,128] only ConvLSTM with dropout = 0.2


The same with no dropout:

out.440443 FAILED 
################# ConvLSTM_True Validation Accuracy =  [0.27799999713897705, 0.31299999356269836, 0.3562000095844269, 0.3662000000476837, 0.391400009393692, 0.36500000953674316, 0.38580000400543213, 0.40119999647140503, 0.39980000257492065, 0.3808000087738037, 0.4293999969959259, 0.3930000066757202, 0.35120001435279846, 0.4043999910354614, 0.367000013589859, 0.39259999990463257, 0.3447999954223633, 0.4016000032424927, 0.3885999917984009, 0.3840000033378601, 0.38940000534057617, 0.38280001282691956, 0.35839998722076416, 0.37380000948905945, 0.35839998722076416, 0.4009999930858612, 0.31220000982284546, 0.3718000054359436, 0.3407999873161316, 0.3571999967098236, 0.326200008392334, 0.32019999623298645, 0.3253999948501587, 0.3416000008583069, 0.3142000138759613, 0.3571999967098236, 0.34439998865127563, 0.3393999934196472, 0.3400000035762787, 0.34060001373291016, 0.3125999867916107, 0.29840001463890076, 0.3375999927520752, 0.3328000009059906, 0.30079999566078186, 0.31779998540878296, 0.3384000062942505, 0.3287999927997589, 0.2985999882221222, 0.32919999957084656, 0.29319998621940613, 0.28200000524520874, 0.28619998693466187, 0.31859999895095825, 0.30140000581741333, 0.3222000002861023, 0.29660001397132874, 0.30320000648498535, 0.26899999380111694, 0.28700000047683716, 0.2587999999523163, 0.31200000643730164, 0.2759999930858612, 0.3230000138282776, 0.313400000333786, 0.3107999861240387, 0.29120001196861267, 0.30219998955726624, 0.3197999894618988, 0.2863999903202057, 0.32120001316070557, 0.25859999656677246, 0.33399999141693115, 0.28439998626708984, 0.28360000252723694, 0.2720000147819519, 0.25619998574256897, 0.29919999837875366, 0.3025999963283539, 0.3206000030040741, 0.2775999903678894, 0.25699999928474426, 0.2955999970436096, 0.2549999952316284, 0.2515999972820282, 0.2856000065803528, 0.27219998836517334, 0.274399995803833, 0.24619999527931213, 0.290800005197525, 0.2648000121116638, 0.289000004529953, 0.24779999256134033, 0.23639999330043793, 0.25360000133514404, 0.2621999979019165, 0.2773999869823456, 0.24979999661445618, 0.2703999876976013, 0.25279998779296875, 0.22179999947547913, 0.2685999870300293, 0.2786000072956085, 0.2248000055551529, 0.2298000007867813, 0.251800000667572, 0.21960000693798065, 0.24979999661445618, 0.2777999937534332, 0.24959999322891235, 0.25859999656677246, 0.2596000134944916, 0.23800000548362732, 0.23720000684261322, 0.2809999883174896, 0.27239999175071716, 0.24480000138282776, 0.2935999929904938, 0.24120000004768372, 0.2558000087738037, 0.290800005197525, 0.2630000114440918, 0.25220000743865967, 0.2651999890804291, 0.2856000065803528, 0.2556000053882599, 0.26080000400543213, 0.22759999334812164, 0.2386000007390976, 0.2531999945640564, 0.22599999606609344, 0.2362000048160553, 0.24500000476837158, 0.23499999940395355, 0.23960000276565552, 0.2565999925136566, 0.2556000053882599, 0.23639999330043793, 0.25380000472068787, 0.2702000141143799, 0.23360000550746918, 0.23280000686645508, 0.23399999737739563, 0.25040000677108765, 0.24779999256134033, 0.24240000545978546, 0.241799995303154, 0.25679999589920044, 0.2624000012874603, 0.23659999668598175, 0.23559999465942383, 0.22579999268054962, 0.23919999599456787, 0.2508000135421753, 0.2362000048160553, 0.23319999873638153, 0.2558000087738037, 0.23499999940395355, 0.2409999966621399, 0.25459998846054077, 0.2378000020980835, 0.23819999396800995, 0.22259999811649323, 0.23819999396800995, 0.22920000553131104, 0.24560000002384186, 0.2362000048160553, 0.24120000004768372, 0.23280000686645508, 0.24160000681877136, 0.23880000412464142, 0.23440000414848328, 0.2401999980211258, 0.23659999668598175, 0.23999999463558197, 0.2305999994277954, 0.24420000612735748, 0.22579999268054962, 0.2574000060558319, 0.2321999967098236, 0.21739999949932098, 0.25600001215934753, 0.23080000281333923, 0.22779999673366547, 0.2134000062942505, 0.24699999392032623, 0.25440001487731934, 0.227400004863739, 0.2117999941110611, 0.20759999752044678, 0.22040000557899475, 0.227400004863739, 0.2329999953508377, 0.2176000028848648, 0.21040000021457672, 0.21539999544620514, 0.2125999927520752, 0.23399999737739563, 0.24320000410079956, 0.22040000557899475]
################# ConvLSTM_True Training Accuracy =  [0.23399999737739563, 0.32268887758255005, 0.3631777763366699, 0.3869555592536926, 0.3992222249507904, 0.4161333441734314, 0.42755556106567383, 0.4332444369792938, 0.4379555583000183, 0.4461555480957031, 0.45153334736824036, 0.4575333297252655, 0.46566668152809143, 0.46566668152809143, 0.4700888991355896, 0.47011110186576843, 0.4773111045360565, 0.4821110963821411, 0.48328888416290283, 0.4873333275318146, 0.4834222197532654, 0.48837777972221375, 0.4940222203731537, 0.4964222311973572, 0.4942888915538788, 0.500177800655365, 0.5001999735832214, 0.5067777633666992, 0.5090888738632202, 0.5114444494247437, 0.5095333456993103, 0.5134222507476807, 0.5151333212852478, 0.5162444710731506, 0.5152444243431091, 0.5188444256782532, 0.5240222215652466, 0.5230000019073486, 0.523711085319519, 0.5237777829170227, 0.5264666676521301, 0.5278666615486145, 0.5301777720451355, 0.5311999917030334, 0.5348666906356812, 0.5371555685997009, 0.5389778017997742, 0.5367555618286133, 0.5401777625083923, 0.5396222472190857, 0.5396000146865845, 0.5420666933059692, 0.5473333597183228, 0.5471333265304565, 0.5472444295883179, 0.548977792263031, 0.5497778058052063, 0.550599992275238, 0.5520889163017273, 0.557200014591217, 0.5553777813911438, 0.55804443359375, 0.5574444532394409, 0.5611777901649475, 0.5606444478034973, 0.5629333257675171, 0.5619555711746216, 0.5598222017288208, 0.5668444633483887, 0.5672222375869751, 0.5652889013290405, 0.5657555460929871, 0.5699999928474426, 0.5727777481079102, 0.5710222125053406, 0.5730666518211365, 0.5738222002983093, 0.5766888856887817, 0.5798666477203369, 0.57833331823349, 0.578000009059906, 0.581933319568634, 0.5794444680213928, 0.5809777975082397, 0.5845555663108826, 0.5846666693687439, 0.5840444564819336, 0.5877777934074402, 0.5886444449424744, 0.5906888842582703, 0.5889999866485596, 0.5908889174461365, 0.5917555689811707, 0.5950888991355896, 0.5946666598320007, 0.5987777709960938, 0.5983777642250061, 0.5985555648803711, 0.6033111214637756, 0.6024222373962402, 0.6026889085769653, 0.6013777852058411, 0.6039555668830872, 0.6026222109794617, 0.6087777614593506, 0.6052666902542114, 0.610622227191925, 0.6129999756813049, 0.6125777959823608, 0.6089777946472168, 0.6111778020858765, 0.6140666604042053, 0.6123777627944946, 0.615066647529602, 0.6156888604164124, 0.6140444278717041, 0.6200888752937317, 0.6220444440841675, 0.618066668510437, 0.6217555403709412, 0.6237555742263794, 0.6195777654647827, 0.6235555410385132, 0.6284444332122803, 0.6275333166122437, 0.6274444460868835, 0.6284666657447815, 0.6295333504676819, 0.6306666731834412, 0.6277555823326111, 0.6322000026702881, 0.6312222480773926, 0.6315333247184753, 0.6285777688026428, 0.6341999769210815, 0.6351555585861206, 0.6378222107887268, 0.6373555660247803, 0.640844464302063, 0.6389777660369873, 0.6407333612442017, 0.6389777660369873, 0.641688883304596, 0.642799973487854, 0.6423110961914062, 0.6447333097457886, 0.6434222459793091, 0.6460000276565552, 0.6466444730758667, 0.6464222073554993, 0.6493777632713318, 0.6493555307388306, 0.6454889178276062, 0.6473110914230347, 0.6494888663291931, 0.6501111388206482, 0.6499999761581421, 0.6497777700424194, 0.6547555327415466, 0.6515111327171326, 0.6538666486740112, 0.6556888818740845, 0.6541555523872375, 0.6547999978065491, 0.6576666831970215, 0.6564666628837585, 0.6586222052574158, 0.6603333353996277, 0.6599777936935425, 0.6605777740478516, 0.6585555672645569, 0.6590444445610046, 0.6628000140190125, 0.6595333218574524, 0.6624000072479248, 0.6612889170646667, 0.6618666648864746, 0.6650888919830322, 0.6636666655540466, 0.6675999760627747, 0.6665777564048767, 0.667377769947052, 0.6673555374145508, 0.6704888939857483, 0.6665777564048767, 0.6667555570602417, 0.6710666418075562, 0.6704221963882446, 0.6684444546699524, 0.670711100101471, 0.6687999963760376, 0.6734222173690796, 0.6697555780410767, 0.6762666702270508, 0.6765999794006348, 0.6751333475112915, 0.6761777997016907, 0.6800888776779175, 0.6782666444778442, 0.6772444248199463]
Trying a smaller network [64,64]


added regular cnn after the convlstm 
and that the dept of the convlstm and cnn = hidden_size, out.984250

'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker')
sys.path.insert(1, '/home/orram/Documents/GitHub/imagewalker')

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

import importlib
importlib.reload(misc)
from misc import Logger
import os 


def deploy_logs():
    if not os.path.exists(hp.save_path):
        os.makedirs(hp.save_path)

    dir_success = False
    for sfx in range(1):  # todo legacy
        candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
        if not os.path.exists(candidate_path):
            hp.this_run_path = candidate_path
            os.makedirs(hp.this_run_path)
            dir_success = Truecnn_net = cnn_one_img(n_timesteps = sample, input_size = 28, input_dim = 1)
            break
    if not dir_success:
        error('run name already exists!')

    sys.stdout = Logger(hp.this_run_path+'log.log')
    print('results are in:', hp.this_run_path)
    print('description: ', hp.description)
    #print('hyper-parameters (partial):', hp.dict)

kernel_regularizer_list = [None, keras.regularizers.l1(),keras.regularizers.l2(),keras.regularizers.l1_l2()]
optimizer_list = [tf.keras.optimizers.Adam, tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop]
if len(sys.argv) > 1:
    paramaters = {
    'epochs' : int(sys.argv[1]),
    
    'sample' : int(sys.argv[2]),
    
    'res' : int(sys.argv[3]),
    
    'hidden_size' : int(sys.argv[4]),
    
    'concat' : int(sys.argv[5]),
    
    'regularizer' : kernel_regularizer_list[int(sys.argv[6])],
    
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

def convgru(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 3, concat = False,
            optimizer = tf.keras.optimizers.Adam, lr = 5e-4):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    
    # define LSTM model
    x = keras.layers.ConvLSTM2D(cell_size, 3, return_sequences=True,padding = 'same',
                                dropout = cnn_dropout, recurrent_dropout=rnn_dropout,
                                kernel_regularizer=regularizer,)(inputA)
    #x = keras.layers.ConvLSTM2D(32, 3, return_sequences=True, padding = 'valid')(x)
    print(x.shape)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(cell_size,(3,3),activation='relu', padding = 'same'))(x)
    x=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x)
    print(x.shape)
    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
    print(x.shape)
    if concat:
        x = keras.layers.Concatenate()([x,inputB])
    print(x.shape)
    x = keras.layers.Flatten()(x)
    print(x.shape)
    x = keras.layers.Dense(1024,activation="relu")(x)
    print(x.shape)
    x = keras.layers.Dense(10,activation="softmax")(x)
    print(x.shape)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'one_layer_convlstm_{}'.format(concat))
    opt=optimizer(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = convgru(n_timesteps = sample, 
                  cell_size = hidden_size,
                  input_size = res,
                  concat = concat,
                  optimizer = optimizer,
                  )

#%%
train_dataset, test_dataset = create_cifar_dataset(images, labels,res = res,
                                    sample = sample, return_datasets=True, 
                                    mixed_state = False, add_seed = 0,
                                    )#bad_res_func = bad_res101, up_sample = True)

train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)

#%%
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


print('################# {} Validation Accuracy = '.format(rnn_net.name),rnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(rnn_net.name),rnn_history.history['sparse_categorical_accuracy'])


plt.figure()
plt.plot(rnn_history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(rnn_history.history['val_sparse_categorical_accuracy'], label = 'val')
plt.legend()
plt.grid()
plt.ylim(0.5,0.7)
plt.title('{} on cifar res = {} hs = {} dropout = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout))
plt.savefig('{} on Cifar res = {} val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}_{}'.format(rnn_net.name, run_id), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
dataset_update(rnn_history, rnn_net,paramaters)    
write_to_file(rnn_history, rnn_net,paramaters)    