from misc import HP
import argparse
import random
import time
import pickle
import copy
import SYCLOP_env as syc
from misc import *
import sys
import os
import cv2
import argparse
import tensorflow.keras as keras

from keras_networks import rnn_model_102
from curriculum_utils import create_mnist_dataset, bad_res102

#scheduler function for curriculum:
class Scheduler():
    def __init__(self,schedule):
        self.schedule = schedule

    def step(self, step, return_val=True):
        # val schedule has a form [(step0,val0),(step1, val1)....(step_last,val_last)]
        # in between the steps a piece-wise linear interpolation is assumed
        if step < self.schedule[0][0]:
            self.val_segment = -1
            self.next_segment_start_step = self.schedule[0][0]
            self.val = self.schedule[0][1]
            self.val_increment = 0
        if step >= self.next_segment_start_step:
            self.val_segment += 1
            if self.val_segment + 1 < len(self.schedule):
                self.next_segment_start_step = self.schedule[self.val_segment + 1][0]
                self.next_segment_start_val = self.schedule[self.val_segment + 1][1]
                self.val_increment = (self.next_segment_start_val - self.val) / (self.next_segment_start_step - step)
            else:
                self.val = self.schedule[-1][1]
                self.next_segment_start_step = 1e30
                self.val_increment = 0
        self.val += self.val_increment
        if return_val:
            return self.val

def deploy_logs():
    if not os.path.exists(hp.save_path):
        os.makedirs(hp.save_path)

    dir_success = False
    for sfx in range(1):  # todo legacy
        candidate_path = hp.save_path + '/' + hp.this_run_name + '_' + str(os.getpid()) + '/'
        if not os.path.exists(candidate_path):
            hp.this_run_path = candidate_path
            os.makedirs(hp.this_run_path)
            dir_success = True
            break
    if not dir_success:
        error('run name already exists!')

    sys.stdout = Logger(hp.this_run_path+'log.log')
    print('results are in:', hp.this_run_path)
    print('description: ', hp.description)
    print('hyper-parameters (partial):', hp.__dict__)

def split_dataset_xy(dataset):
    dataset_x1 = [uu[0] for uu in dataset]
    dataset_x2 = [uu[1] for uu in dataset]
    dataset_y = [uu[-1] for uu in dataset]
    return (np.array(dataset_x1)[...,np.newaxis],np.array(dataset_x2)[:,:n_timesteps,:]),np.array(dataset_y)

#parse hyperparameters

lsbjob = os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob

hp = HP()
hp.save_path = 'saved_runs'
hp.description=''
parser = argparse.ArgumentParser()
parser.add_argument('--tau_int', default=4., type=float, help='Integration timescale for adaaptation')
parser.add_argument('--resize', default=1.0, type=float, help='resize of images')
parser.add_argument('--run_name_suffix', default='', type=str, help='suffix for runname')
parser.add_argument('--eval_dir', default=None, type=str, help='eval dir')

parser.add_argument('--dqn_initial_network', default=None, type=str, help='dqn_initial_network')
parser.add_argument('--decoder_initial_network', default=None, type=str, help='decoder_initial_network')
parser.add_argument('--decoder_learning_rate',  default=1e-3, type=float, help='decoder learning rate')
parser.add_argument('--decoder_dropout',  default=0.0, type=float, help='decoder dropout')
parser.add_argument('--decoder_rnn_type',  default='gru', type=str, help='gru or rnn')
parser.add_argument('--decoder_rnn_units',  default=100, type=int, help='decoder rnn units')
parser.add_argument('--decoder_rnn_layers',  default=1, type=int, help='decoder rnn units')


parser.add_argument('--decoder_ignore_position', dest='decoder_ignore_position', action='store_true')
parser.add_argument('--no-decoder_ignore_position', dest='decoder_ignore_position', action='store_false')

parser.add_argument('--syclop_learning_rate',  default=2.5e-3, type=float, help='syclop (RL) learning rate')

parser.add_argument('--color', default='grayscale', type=str, help='grayscale/rgb')
parser.add_argument('--speed_reward',  default=0.0, type=float, help='speed reward, typically negative')
parser.add_argument('--intensity_reward',  default=0.0, type=float, help='speed penalty reward')
parser.add_argument('--loss_reward',  default=-1.0, type=float, help='reward for loss, typically negative')
parser.add_argument('--resolution',  default=28, type=int, help='resolution')
parser.add_argument('--max_eval_episodes',  default=10000, type=int, help='episodes for evaluation mode')
parser.add_argument('--steps_per_episode',  default=5, type=int, help='time steps in each episode in ')
parser.add_argument('--fit_verbose',  default=1, type=int, help='verbose level for model.fit                        ')
parser.add_argument('--steps_between_learnings',  default=100, type=int, help='steps_between_learnings')
parser.add_argument('--num_epochs',  default=100, type=int, help='steps_between_learnings')

parser.add_argument('--alpha_increment',  default=0.01, type=float, help='reward for loss, typically negative')


parser.add_argument('--beta_t1',  default=400000, type=int, help='time rising bete')
parser.add_argument('--beta_t2',  default=700000, type=int, help='end rising beta')
parser.add_argument('--beta_b1',  default=0.1, type=float, help='beta initial value')
parser.add_argument('--beta_b2',  default=1.0, type=float, help='beta final value')

parser.add_argument('--curriculum_enable', dest='curriculum_enable', action='store_true')
parser.add_argument('--no-curriculum_enable', dest='curriculum_enable', action='store_false')

parser.add_argument('--conv_fe', dest='conv_fe', action='store_true')
parser.add_argument('--no-conv_fe', dest='conv_fe', action='store_false')


parser.set_defaults(eval_mode=False, decode_from_dvs=False,test_mode=False,rising_beta_schedule=True,decoder_ignore_position=False, curriculum_enable=True, conv_fe=False)

config = parser.parse_args()
config = vars(config)
hp.upadte_from_dict(config)
hp.this_run_name = sys.argv[0] + '_noname_' + hp.run_name_suffix + '_' + lsbjob + '_' + str(int(time.time()))

#define model
n_timesteps = hp.steps_per_episode

##
deploy_logs()
##
decoder = rnn_model_102(lr=hp.decoder_learning_rate,ignore_input_B=hp.decoder_ignore_position,dropout=hp.decoder_dropout,rnn_type=hp.decoder_rnn_type,
                                input_size=(hp.resolution,hp.resolution, 1),rnn_layers=hp.decoder_rnn_layers,conv_fe=hp.conv_fe)
#define dataset
(images, labels), (images_test, labels_test) = keras.datasets.mnist.load_data(path="mnist.npz")


#fit one epoch in a  time
# scheduler = Scheduler(hp.lambda_schedule)
# for epoch in range(hp.num_epochs):
#     lambda_epoch = scheduler.step(epoch)
alpha=0
for epoch in range(hp.num_epochs):
    if hp.curriculum_enable:
        if epoch == 0:
            train_dataset, test_dataset = create_mnist_dataset(images, labels, 6, bad_res_func=bad_res102, return_datasets=True, q_0=0, alpha=1.0,random_trajectories=False)
            train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
            test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
            q_0=train_dataset_x[1][0]
            print('debu0',q_0)
        else:
            alpha += hp.alpha_increment
            alpha = np.minimum(alpha,1.0)

            train_dataset, test_dataset = create_mnist_dataset(images, labels, 6, bad_res_func=bad_res102, return_datasets=True, q_0=q_0, alpha=alpha,random_trajectories=True)
            train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
            test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
            q_prime=train_dataset_x[1][0]
            print('epoch',epoch,' alpha',alpha,'first q --', q_prime.reshape([-1]))
    else:
        train_dataset, test_dataset = create_mnist_dataset(images, labels, 6, bad_res_func=bad_res102,
                                                           return_datasets=True, q_0=0, alpha=1.0,
                                                           random_trajectories=True)
        train_dataset_x, train_dataset_y = split_dataset_xy(train_dataset)
        test_dataset_x, test_dataset_y = split_dataset_xy(test_dataset)
        q_prime = train_dataset_x[1][0]
        print('epoch', epoch, '  CONTROL!!!',' first q --', q_prime.reshape([-1]))

    print("Fit model on training data")
    history = decoder.fit(
        train_dataset_x,
        train_dataset_y,
        batch_size=64,
        epochs=1,
        verbose=hp.fit_verbose,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(test_dataset_x, test_dataset_y)  # (validation_images, validation_labels)
        )

#save the model
decoder.save(hp.this_run_path + 'final_decoder.nwk')
print('results are in:', hp.this_run_path)
