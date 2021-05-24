                                                                                                                                                                                                                                                                                                                                        #from image_env_mnist1 import Image_env1
from RL_brain_c import DeepQNetwork
import numpy as np
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
from keras_networks import rnn_model_101


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


def local_observer(sensor,agent):
    normfactor=1.0/256.0
    return normfactor*np.concatenate([relu_up_and_down(sensor.central_dvs_view),
            relu_up_and_down(cv2.resize(1.0*sensor.dvs_view, dsize=(16, 16), interpolation=cv2.INTER_AREA))])
#Define function for low resolution lens on syclop

def bad_res101(img,res):
    sh=np.shape(img)
    dwnsmp=cv2.resize(img,res, interpolation = cv2.INTER_CUBIC)
    upsmp = cv2.resize(dwnsmp,sh, interpolation = cv2.INTER_CUBIC)
    return upsmp

def prep_singeltons_for_feed(items):
    items_=[]
    for item in items:
        shp=list(np.shape(item))
        items_.append(np.reshape(item,[1]+shp))
    return items_

def run_env(eval_mode=False,return_actions=False,return_positions=False,forced_actions=None,forced_positions=None):
    old_policy_map=0
    step = 0
    batch_acc_list=[]
    best_thus_far = -1e10
    running_ave_reward = 0
    running_ave_records = 0
    if return_actions:
        action_rec=[]
    if return_positions:
        position_rec=[]

    imim_list = []
    pospos_list = []
    lbl_list = []
    for episode in range(hp.max_episode):
        if return_actions:
            episode_action_rec = []
        if return_positions:
            episode_position_rec = []
        observation = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        observation_ = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        # scene.current_frame = np.random.choice(scene.total_frames)
        imgndx=episode % len(images)
        img = 256* build_mnist_padded(1. / 256 * np.reshape(images[imgndx], [1, 28, 28])) #todo
        lbl = labels[imgndx]
        scene.image = img

        agent.reset(centered=True)
        #decoder related stuff
        imim=[]
        pospos=[]
        ##
        if forced_positions is not None:
            agent.set_manual_trajectory(forced_positions[episode])
            agent.manual_act()
        sensor.reset()
        sensor.update(scene, agent)
        sensor.update(scene, agent)
        for step_prime in range(hp.steps_per_episode):

            if hp.decode_from_dvs:
                imim.append(sensor.central_dvs_view[...,np.newaxis]+0.)
            else:
                imim.append(sensor.central_frame_view[...,np.newaxis]+0.)

            pospos.append(agent.q+0.)
            if len(imim) == hp.steps_per_episode:
                this_loss,aa = decoder.test_on_batch(
                            prep_singeltons_for_feed([imim,pospos]), y=np.reshape(lbl,[1,1]))
                batch_acc_list.append(aa+0.)
                imim_list.append(imim)
                pospos_list.append(pospos)
                lbl_list.append(lbl)
                # if eval_mode:
                #     print('positions / accuracy', pospos,aa)
                # print('loss,acc',this_loss, aa)
            else:
                this_loss, aa = (0.,0.)
            if forced_actions is not None:
                action = forced_actions[episode][step_prime]
            elif forced_positions is not None:
                pass
            else:
                action, ent = RL.choose_action(observation.reshape([-1]))

                ent = \
                    ent if not np.isnan(ent) else 0
                running_ave_records = 0.999*running_ave_records + 0.001*ent
                # print('ent:',ent)
            if return_actions:
                episode_action_rec.append(action)
            if return_positions:
                episode_position_rec.append(agent.q)
            reward.update_rewards(sensor = sensor, agent = agent, manual_reward=this_loss)

            running_ave_reward = 0.999*running_ave_reward+0.001*np.array([reward.reward]+reward.rewards.tolist())

            if step % 10000 < 1000:
                # print([agent.q_ana[0], agent.q_ana[1], reward.reward] , reward.rewards , [RL.epsilon])
                # print(type([agent.q_ana[0], agent.q_ana[1], reward.reward]) , type(reward.rewards), type([RL.epsilon]))
                recorder.record([agent.q_ana[0],agent.q_ana[1],reward.reward]+reward.rewards.tolist()+[RL.epsilon])
            if forced_positions is None:
                agent.act(action)
            else:
                agent.manual_act()
            sensor.update(scene,agent)
            # scene.update()
            observation_ *= hp.fading_mem
            observation_ += local_observer(sensor, agent)  # todo: generalize

            ## learning related stuff
            if not eval_mode:
                RL.store_transition(observation.reshape([-1]), action, reward.reward, observation_.reshape([-1]))
            observation = copy.copy(observation_)
            step += 1
            if (step > 100) and (step % hp.steps_between_learnings == 0):
                if (not eval_mode) and (not hp.test_mode):
                    RL.learn()

            if step%1000 ==0:
                print(episode,step,' running reward   ',running_ave_reward)
                print('entropy etc.:', running_ave_records)
                # print('frame:', scene.current_frame,)
                if not eval_mode:
                    if running_ave_reward[0] > best_thus_far:
                        best_thus_far = running_ave_reward[0]
                        RL.dqn.save_nwk_param(hp.this_run_path+'best_liron.nwk')
                        decoder.save(hp.this_run_path + 'best_decoder.nwk')
                        print('saved best network, mean reward: ', best_thus_far)
            if step%10000 ==0:
                    recorder.plot()
                    if not eval_mode:
                        RL.dqn.save_nwk_param(hp.this_run_path+'tempX_1.nwk')
                        decoder.save(hp.this_run_path + 'tempX_decoder.nwk')

        if return_actions:
            action_rec.append(episode_action_rec)
        if return_positions:
            position_rec.append(episode_position_rec)
        if (episode > 0 and episode % hp.train_decoder_every == 0) and (not eval_mode) and (not hp.test_mode):
            # print('debug shapes:', np.shape(imim_list))
            # print('debug shapes:',np.shape(pospos_list))
            # print('debug shapes:', np.shape(lbl_list))
            tr_loss,tr_acc = decoder.train_on_batch([np.array(imim_list), np.array(pospos_list)], y=np.array(lbl_list))
            print('training step:', tr_loss,tr_acc)
            imim_list=[]
            pospos_list =[]
            lbl_list =[]

        if (episode > 0 and episode % hp.report_accuracy_every == 0):
            print('running accuracy  ', np.mean(batch_acc_list))
            batch_acc_list = []

            # debug_policy_plot()
            # if step % 100000 == 0:
            #         recorder.save(hp.this_run_path+recorder_file)
    if return_actions and return_positions:
        return action_rec, position_rec
    if return_actions:
        return action_rec
    if return_positions:
        return position_rec
    # recorder.save(hp.this_run_path+recorder_file)


if __name__ == "__main__":

    lsbjob = os.getenv('LSB_JOBID')
    lsbjob = '' if lsbjob is None else lsbjob

    hp = HP()
    hp.save_path = 'saved_runs'

    # hp.description = "only 2nd image from videos 1st frame, penalty for speed, soft q learning"
    hp.description = "hyperacuity games"
    hp.mem_depth = 1
    hp.padding = [32, 32]
    hp.max_episode = 10000
    hp.steps_per_episode = 100
    hp.steps_between_learnings = 100
    hp.steps_before_learning_begins = 100
    hp.saccade_observation_scale = 100
    hp.fading_mem = 0.0
    hp.drift_signal_to_saccade_en = 1
    hp.fisheye_file =  None #'fisheye_102.pkl'
    recorder_file = 'records.pkl'
    hp_file = 'hp.pkl'
    hp.speed_reward = 0
    hp.dqn_initial_network = None  # 'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1590422112_0/tempX_1.nwk'  # None #'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1589383850_0/tempX_1.nwk'
    hp.eval_mode = False
    hp.eval_dirs = []
    # hp.eval_dirs =['saved_runs/run_saccader103.py_noname__tau20__1609610126__70900/','saved_runs/run_saccader103.py_noname__hello3__1609401198__31923/']

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

    parser.add_argument('--decoder_ignore_position', dest='decoder_ignore_position', action='store_true')
    parser.add_argument('--no-decoder_ignore_position', dest='decoder_ignore_position', action='store_false')

    parser.add_argument('--syclop_learning_rate',  default=2.5e-3, type=float, help='syclop (RL) learning rate')

    parser.add_argument('--color', default='grayscale', type=str, help='grayscale/rgb')
    parser.add_argument('--speed_reward',  default=0.0, type=float, help='speed reward, typically negative')
    parser.add_argument('--intensity_reward',  default=0.0, type=float, help='speed penalty reward')
    parser.add_argument('--loss_reward',  default=-1.0, type=float, help='reward for loss, typically negative')
    parser.add_argument('--resolution',  default=28, type=int, help='resolution')
    parser.add_argument('--max_eval_episodes',  default=10000, type=int, help='episodes for evaluation mode')
    parser.add_argument('--steps_per_episode',  default=100, type=int, help='time steps in each episode in ')
    parser.add_argument('--steps_between_learnings',  default=100, type=int, help='steps_between_learnings')

    parser.add_argument('--beta_t1',  default=400000, type=int, help='time rising bete')
    parser.add_argument('--beta_t2',  default=700000, type=int, help='end rising beta')
    parser.add_argument('--beta_b1',  default=0.1, type=float, help='beta initial value')
    parser.add_argument('--beta_b2',  default=1.0, type=float, help='beta final value')



    parser.add_argument('--train_decoder_every',  default=100, type=int, help='time steps between decoder trainings ')
    parser.add_argument('--max_episode',  default=10000, type=int, help='episodes for training mode')
    parser.add_argument('--report_accuracy_every',  default=10000, type=int, help='report_accuracy_every .. episodes')

    parser.add_argument('--test_mode', dest='test_mode', action='store_true')
    parser.add_argument('--no-test_mode', dest='test_mode', action='store_false')

    parser.add_argument('--eval_mode', dest='eval_mode', action='store_true')
    parser.add_argument('--no-eval_mode', dest='eval_mode', action='store_false')

    parser.add_argument('--rising_beta_schedule', dest='rising_beta_schedule', action='store_true')
    parser.add_argument('--no-rising_beta_schedule', dest='rising_beta_schedule', action='store_false')

    parser.add_argument('--decode_from_dvs', dest='decode_from_dvs', action='store_true')
    parser.add_argument('--no-decode_from_dvs', dest='decode_from_dvs', action='store_false')
    parser.set_defaults(eval_mode=False, decode_from_dvs=False,test_mode=False,rising_beta_schedule=True,decoder_ignore_position=False)


    config = parser.parse_args()
    config = vars(config)
    hp.upadte_from_dict(config)
    #finish with updates
    if hp.fisheye_file is None:
        fy_dict = None
    else:
        with open(hp.fisheye_file, 'rb') as f:
            fy_dict = pickle.load(f)
    hp.this_run_name = sys.argv[0] + '_noname_' + hp.run_name_suffix + '_' + lsbjob + '_' + str(int(time.time()))
    hp.grayscale = hp.color == 'grayscale'
    nchannels = 1 if hp.grayscale else 3

    recorder = Recorder(n=6)


    # with open('../video_datasets/liron_images/shuffled_images.pkl', 'rb') as f:
    #     images = pickle.load(f)

    ##
    # mnist = MNIST('/home/bnapp/datasets/mnist/')
    # images, labels = mnist.load_training()
    (images, labels), (images_test,labels_test) = keras.datasets.mnist.load_data(path="mnist.npz")
    if hp.test_mode or hp.eval_mode:
        (images, labels) = (images_test, labels_test)
    img = build_mnist_padded(1. / 256 * np.reshape(images[0], [1, 28, 28])) #just to initialize scene with correct size

    scene = syc.Scene(image_matrix=img)
    sensor = syc.Sensor(winx=56, winy=56, centralwinx=28, centralwiny=28)
    sensor.hp.resolution_fun = lambda x: bad_res101(x, (hp.resolution, hp.resolution))
    agent = syc.Agent(max_q=[scene.maxx - sensor.hp.winx, scene.maxy - sensor.hp.winy])

    reward = syc.Rewards(reward_types=['central_rms_intensity', 'speed','manual_reward'],relative_weights=[hp.intensity_reward,hp.speed_reward,hp.loss_reward])
    # observation_size = sensor.hp.winx*sensor.hp.winy*2
    observation_size = 2080

    rising_beta_schedule = [[hp.beta_t1 // hp.steps_between_learnings, hp.beta_b1], [hp.beta_t2 // hp.steps_between_learnings, hp.beta_b2]]
    flat_beta_schedule = [[hp.beta_t1 // hp.steps_between_learnings, hp.beta_b2], [hp.beta_t2 // hp.steps_between_learnings, hp.beta_b2]]

    # rising_beta_schedule = [[400000 // hp.steps_between_learnings, 0.1], [700000 // hp.steps_between_learnings, 1]]
    # flat_beta_schedule = [[400000 // hp.steps_between_learnings, 1.0], [700000 // hp.steps_between_learnings, 1]]

    RL = DeepQNetwork(len(agent.hp.action_space), observation_size*hp.mem_depth,#sensor.frame_size+2,
                      reward_decay=0.99,
                      e_greedy=0.95,
                      e_greedy0=0.8,
                      replace_target_iter=10,
                      memory_size=100000,
                      e_greedy_increment=0.0001,
                      learning_rate=hp.syclop_learning_rate,
                      double_q=True,
                      dqn_mode=True,
                      state_table=np.zeros([1,observation_size*hp.mem_depth]),
                      soft_q_type='boltzmann',
                      beta_schedule=rising_beta_schedule if hp.rising_beta_schedule else flat_beta_schedule,
                      arch='mlp')
    keras.backend.set_session(RL.dqn.sess)
    if hp.decoder_initial_network is None:
        decoder = rnn_model_101(lr=hp.decoder_learning_rate,ignore_input_B=hp.decoder_ignore_position,dropout=hp.decoder_dropout,rnn_type=hp.decoder_rnn_type)
    else:
        decoder = keras.models.load_model(hp.decoder_initial_network) #for example: 'ref_nets/keras_decoder_5stp_101.model'
        keras.backend.set_value(decoder.optimizer.lr, hp.decoder_learning_rate)
    if not(hp.dqn_initial_network is None):
        RL.dqn.load_nwk_param(hp.dqn_initial_network)
    hp.scene = scene.hp
    # hp.sensor = sensor.hp #todo, function is now a part of hp...
    hp.agent = agent.hp
    hp.reward = reward.hp
    hp.RL = RL.hp
    deploy_logs()
    if not hp.eval_mode:
        with open(hp.this_run_path+hp_file, 'wb') as f:
            pickle.dump(hp, f)
    if not hp.eval_mode:
        run_env()
    else:
        hp.max_episode = hp.max_eval_episodes
        # if hp.eval_dir is not None:
        #     hp.eval_dirs = [hp.eval_dir]
        #for run_dir in hp.eval_dirs:
        if True:
            # RL.dqn.load_nwk_param(run_dir+'tempX_1.nwk')
            # hp.update_attributes_from_file(run_dir+'hp.pkl',['tau_int'])
            # print('Testing results from:', run_dir)
            # RL.beta=1e-10
            # print('Setting random policy, beta=',RL.beta)
            # run_env(eval_mode=True)
            RL.beta=1.
            print('Setting large beta policy, beta=',RL.beta)
            act_rec,pos_rec=run_env(eval_mode=True,return_actions=True,return_positions=True)
            print('re-running with forced actions')
            run_env(eval_mode=True, forced_actions=act_rec)
            # print('re-running with forced positions')
            # run_env(eval_mode=True, forced_positions=pos_rec)
            print('re-running with forced actions, shuffled')
            random.shuffle(act_rec)
            run_env(eval_mode=True, forced_actions=act_rec)
            # with open(hp.this_run_path + 'position_records.pkl', 'wb') as f:
            #     pickle.dump(pos_rec, f)
    print('results are in:', hp.this_run_path)

#saved_runs/run_syclop_resolution101.py_noname__1619022130__111368/ -10
