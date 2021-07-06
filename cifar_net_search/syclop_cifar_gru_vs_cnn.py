'''
I changed the cnn so it takes a random index from the syclop samples to see
if it bassicaly is equal to data augmentation.

'epochs': 30, 'sample': 20, 'res': 8, 'hidden_size': 128, 'concat': 1, 
'regularizer': None, 'optimizer': Adam 'cnn_dropout': 0.4, 'rnn_dropout': 0.2, 'lr': 0.0005, 'run_id': 6557
################# extended_cnn_one_img Validation Accuracy =  [0.3100000023841858, 0.352400004863739, 0.37599998712539673, 0.3901999890804291, 0.3970000147819519, 0.3986000120639801, 0.3984000086784363, 0.4163999855518341, 0.4212000072002411, 0.42100000381469727, 0.43160000443458557, 0.42399999499320984, 0.4300000071525574, 0.4323999881744385, 0.4341999888420105, 0.4318000078201294, 0.44339999556541443, 0.44940000772476196, 0.4456000030040741, 0.45239999890327454, 0.44940000772476196, 0.44760000705718994, 0.45739999413490295, 0.44780001044273376, 0.4546000063419342, 0.44519999623298645, 0.45899999141693115, 0.45840001106262207, 0.460999995470047, 0.44999998807907104, 0.46219998598098755, 0.4641999900341034, 0.4708000123500824, 0.46239998936653137, 0.462799996137619, 0.4675999879837036, 0.46480000019073486, 0.46380001306533813, 0.460999995470047, 0.46779999136924744, 0.46939998865127563, 0.47099998593330383, 0.4657999873161316, 0.4659999907016754, 0.46779999136924744, 0.46860000491142273, 0.4717999994754791, 0.4745999872684479, 0.47360000014305115, 0.46939998865127563, 0.4690000116825104, 0.4787999987602234, 0.46860000491142273, 0.4593999981880188, 0.4747999906539917, 0.462799996137619, 0.4708000123500824, 0.47540000081062317, 0.4729999899864197, 0.4713999927043915, 0.48260000348091125, 0.46959999203681946, 0.47600001096725464, 0.4724000096321106, 0.4740000069141388, 0.4691999852657318, 0.48100000619888306, 0.47279998660087585, 0.4740000069141388, 0.4675999879837036, 0.4699999988079071, 0.46540001034736633, 0.4650000035762787, 0.477400004863739, 0.47679999470710754, 0.4733999967575073, 0.4684000015258789, 0.47620001435279846, 0.47099998593330383, 0.4758000075817108, 0.4749999940395355, 0.4740000069141388, 0.4763999879360199, 0.47360000014305115, 0.48179998993873596, 0.47620001435279846, 0.47440001368522644, 0.4742000102996826, 0.4805999994277954, 0.47940000891685486, 0.47699999809265137, 0.4819999933242798, 0.4828000068664551, 0.4715999960899353, 0.47699999809265137, 0.4731999933719635, 0.47600001096725464, 0.4772000014781952, 0.48179998993873596, 0.4848000109195709]
################# extended_cnn_one_img Training Accuracy =  [0.23971110582351685, 0.3173777759075165, 0.34351110458374023, 0.35911110043525696, 0.3735555410385132, 0.380355566740036, 0.38946667313575745, 0.39417776465415955, 0.39615556597709656, 0.39986667037010193, 0.40639999508857727, 0.41126665472984314, 0.4132888913154602, 0.4151555597782135, 0.42011111974716187, 0.4196222126483917, 0.4236222207546234, 0.4285777807235718, 0.43033334612846375, 0.4335777759552002, 0.43646666407585144, 0.4375111162662506, 0.43637776374816895, 0.43815556168556213, 0.4420222342014313, 0.444599986076355, 0.44388890266418457, 0.4457777738571167, 0.4440666735172272, 0.4484444558620453, 0.4474888741970062, 0.4498000144958496, 0.4492666721343994, 0.45135554671287537, 0.4537777900695801, 0.4556444585323334, 0.45764443278312683, 0.4585777819156647, 0.4553777873516083, 0.45675554871559143, 0.4603999853134155, 0.4587555527687073, 0.4611110985279083, 0.46433332562446594, 0.4634888768196106, 0.46257779002189636, 0.46746665239334106, 0.4647333323955536, 0.46577778458595276, 0.4673333466053009, 0.469177782535553, 0.4694444537162781, 0.46904444694519043, 0.46622222661972046, 0.46924445033073425, 0.4692666530609131, 0.4700888991355896, 0.47126665711402893, 0.4735333323478699, 0.47171109914779663, 0.4752666652202606, 0.47064444422721863, 0.4732222259044647, 0.4788222312927246, 0.4758000075817108, 0.474355548620224, 0.47833332419395447, 0.4769555628299713, 0.480222225189209, 0.47682222723960876, 0.4770444333553314, 0.474866658449173, 0.47484445571899414, 0.47913333773612976, 0.48295554518699646, 0.4795111119747162, 0.48188889026641846, 0.4798666536808014, 0.4799555540084839, 0.48002222180366516, 0.48091110587120056, 0.4810444414615631, 0.48028889298439026, 0.48019999265670776, 0.4835333228111267, 0.4823555648326874, 0.4805777668952942, 0.48253333568573, 0.48322221636772156, 0.48411110043525696, 0.4854888916015625, 0.48539999127388, 0.4824666678905487, 0.4834444522857666, 0.485577791929245, 0.48444443941116333, 0.4855555593967438, 0.4860000014305115, 0.4836222231388092, 0.4856666624546051]
##################### Fit cnn_gru_1 and trajectories model on training data res = 8 ##################
################# cnn_gru_1 Validation Accuracy =  [0.2685999870300293, 0.3610000014305115, 0.38519999384880066, 0.4004000127315521, 0.42320001125335693, 0.4293999969959259, 0.43779999017715454, 0.42739999294281006, 0.46380001306533813, 0.4537999927997589, 0.4742000102996826, 0.4578000009059906, 0.4862000048160553, 0.48500001430511475, 0.4918000102043152, 0.4970000088214874, 0.4968000054359436, 0.5081999897956848, 0.5144000053405762, 0.5041999816894531, 0.5181999802589417, 0.5242000222206116, 0.5260000228881836, 0.5307999849319458, 0.5270000100135803, 0.5360000133514404, 0.5317999720573425, 0.5447999835014343, 0.5351999998092651, 0.5483999848365784, 0.5320000052452087, 0.5407999753952026, 0.5490000247955322, 0.5501999855041504, 0.5591999888420105, 0.5540000200271606, 0.5544000267982483, 0.5565999746322632, 0.5515999794006348, 0.5546000003814697, 0.5655999779701233, 0.5673999786376953, 0.5626000165939331, 0.5708000063896179, 0.5618000030517578, 0.5662000179290771, 0.5651999711990356, 0.5673999786376953, 0.5699999928474426, 0.5727999806404114, 0.5604000091552734, 0.5601999759674072, 0.5712000131607056, 0.5655999779701233, 0.5723999738693237, 0.5672000050544739, 0.5723999738693237, 0.5712000131607056, 0.5702000260353088, 0.5722000002861023, 0.5756000280380249, 0.5712000131607056, 0.5766000151634216, 0.5763999819755554, 0.5776000022888184, 0.5672000050544739, 0.569599986076355, 0.5763999819755554, 0.5727999806404114, 0.5705999732017517, 0.5622000098228455, 0.5756000280380249, 0.5655999779701233, 0.5752000212669373, 0.5795999765396118, 0.5748000144958496, 0.5753999948501587, 0.5709999799728394, 0.5753999948501587, 0.5723999738693237, 0.5806000232696533, 0.5771999955177307, 0.5691999793052673, 0.576200008392334, 0.5807999968528748, 0.5821999907493591, 0.5767999887466431, 0.5705999732017517, 0.5738000273704529, 0.5774000287055969, 0.5777999758720398, 0.5825999975204468, 0.5663999915122986, 0.5753999948501587, 0.5809999704360962, 0.5759999752044678, 0.5881999731063843, 0.5884000062942505, 0.5748000144958496, 0.5824000239372253]
################# cnn_gru_1 Training Accuracy =  [0.22579999268054962, 0.31679999828338623, 0.3536222279071808, 0.37700000405311584, 0.39482221007347107, 0.40584444999694824, 0.4203111231327057, 0.42775556445121765, 0.4346444308757782, 0.44046667218208313, 0.44708889722824097, 0.45284444093704224, 0.4582222104072571, 0.46788889169692993, 0.47386667132377625, 0.47751110792160034, 0.48517778515815735, 0.4938444495201111, 0.5007110834121704, 0.5042444467544556, 0.5076666474342346, 0.5123111009597778, 0.5171111226081848, 0.5212666392326355, 0.5219777822494507, 0.5269333124160767, 0.5286666750907898, 0.5320888757705688, 0.5338444709777832, 0.5334222316741943, 0.5381110906600952, 0.5425778031349182, 0.5415111184120178, 0.5460888743400574, 0.5490444302558899, 0.5503555536270142, 0.553422212600708, 0.5545777678489685, 0.5555333495140076, 0.5577777624130249, 0.5581333041191101, 0.5604222416877747, 0.562666654586792, 0.5621777772903442, 0.5636000037193298, 0.5645111203193665, 0.5709333419799805, 0.5658000111579895, 0.5680222511291504, 0.5718666911125183, 0.5734444260597229, 0.5727555751800537, 0.5752221941947937, 0.5756666660308838, 0.5758888721466064, 0.5776888728141785, 0.5772666931152344, 0.5799333453178406, 0.5793777704238892, 0.5815555453300476, 0.5828444361686707, 0.585266649723053, 0.5816222429275513, 0.5850666761398315, 0.5868666768074036, 0.5912666916847229, 0.5874888896942139, 0.5864889025688171, 0.5894444584846497, 0.5912666916847229, 0.5914666652679443, 0.5872222185134888, 0.5955333113670349, 0.5932444334030151, 0.5931110978126526, 0.5950666666030884, 0.5959555506706238, 0.5946666598320007, 0.5951777696609497, 0.5996000170707703, 0.5950222015380859, 0.5976444482803345, 0.5952666401863098, 0.5966444611549377, 0.5973333120346069, 0.602400004863739, 0.6047555804252625, 0.6009111404418945, 0.6002222299575806, 0.6012666821479797, 0.6017777919769287, 0.6056222319602966, 0.6058666706085205, 0.6051777601242065, 0.6025111079216003, 0.6047777533531189, 0.6070444583892822, 0.6089777946472168, 0.608488917350769, 0.6082888841629028]

Same with epochs:200 out.981098

Same but changed the dropout of the cnn one img network to 0.2, out.981848

Same dropout = 0.2 and epochs:300 out.981838

Same but changed the dropout of the cnn one img network to 0.2, out.986217

Same dropout = 0.2 and epochs:300 out.
'''

from __future__ import division, print_function, absolute_import

print('Starting..................................')
import os
import sys
sys.path.insert(1, '/home/labs/ahissarlab/orra/imagewalker/')
import numpy as np
import cv2
import misc
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from keras_utils import dataset_update, write_to_file, create_cifar_dataset, extended_cnn_one_img
from misc import *

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
images, labels = trainX, trainy

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

def cnn_gru(n_timesteps = 5, hidden_size = 128,input_size = 32, concat = True, 
            optimizer = tf.keras.optimizers.Adam, ):
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

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same'))(x1)
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
    x = keras.layers.GRU(hidden_size,input_shape=(n_timesteps, None),
                         return_sequences=True,recurrent_dropout=rnn_dropout,
                         kernel_regularizer=regularizer)(x)
    
    x = keras.layers.Flatten()(x)
    #Add another dense layer (prior it reached 62%)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'cnn_gru_{}'.format(concat))
    opt=optimizer(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

rnn_net = cnn_gru(n_timesteps = sample, hidden_size = hidden_size,input_size = res, concat = concat)
cnn_net = cnn_net = extended_cnn_one_img(n_timesteps = sample, input_size = res, dropout = 0.0)


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

print("##################### Fit {} and trajectories model on training data res = {} ##################".format(cnn_net.name,res))
cnn_history = cnn_net.fit(
    train_dataset_x,
    train_dataset_y,
    batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test_dataset_x, test_dataset_y),
    verbose = 0)

print('################# {} Validation Accuracy = '.format(cnn_net.name),cnn_history.history['val_sparse_categorical_accuracy'])
print('################# {} Training Accuracy = '.format(cnn_net.name),cnn_history.history['sparse_categorical_accuracy'])


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
plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
# plt.plot(cnn_history.history['sparse_categorical_accuracy'], label = 'cnn train')
# plt.plot(cnn_history.history['val_sparse_categorical_accuracy'], label = 'cnn val')
plt.legend()
plt.grid()
plt.ylim(0.5,0.7)
plt.title('{} on cifar res = {} hs = {} dropout = {}, num samples = {}'.format(rnn_net.name, res, hidden_size,cnn_dropout,sample))
plt.savefig('{} on Cifar res = {}, rnn vs cnn, val accur = {} hs = {} dropout = {}.png'.format(rnn_net.name,res,rnn_history.history['val_sparse_categorical_accuracy'][-1], hidden_size,cnn_dropout))

with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}'.format(run_id), 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
    
# with open('/home/labs/ahissarlab/orra/imagewalker/cifar_net_search/{}HistoryDict'.format(cnn_net.name), 'wb') as file_pi:
#     pickle.dump(cnn_history.history, file_pi)


dataset_update(rnn_history, rnn_net,paramaters)    
write_to_file(rnn_history, rnn_net,paramaters)    
    
    