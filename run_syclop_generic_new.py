                                                                                                                                                                                                                                                                                                                                        #from image_env_mnist1 import Image_env1
from RL_brain_c import DeepQNetwork
import numpy as np
import time
import pickle
import copy
import SYCLOP_env as syc
from misc import *
import sys
import os
import cv2
import argparse


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
def run_env():
    old_policy_map=0
    step = 0
    best_thus_far = -1e10
    running_ave_reward = 0

    for episode in range(hp.max_episode):
        observation = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        observation_ = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        scene.current_frame = np.random.choice(scene.total_frames)
        scene.image = scene.frame_list[scene.current_frame]

        agent.reset()
        # agent.q_ana[1]=256./2.-32
        # agent.q_ana[0]=192./2-32
        # agent.q = np.int32(np.floor(agent.q_ana))

        sensor.reset()
        sensor.update(scene, agent)
        sensor.update(scene, agent)
        for step_prime in range(hp.steps_per_episode):
            action, ent = RL.choose_action(observation.reshape([-1]))
            reward.update_rewards(sensor = sensor, agent = agent)
            running_ave_reward = 0.999*running_ave_reward+0.001*np.array([reward.reward]+reward.rewards.tolist())
            if step % 10000 < 1000:
                # print([agent.q_ana[0], agent.q_ana[1], reward.reward] , reward.rewards , [RL.epsilon])
                # print(type([agent.q_ana[0], agent.q_ana[1], reward.reward]) , type(reward.rewards), type([RL.epsilon]))
                recorder.record([agent.q_ana[0],agent.q_ana[1],reward.reward]+reward.rewards.tolist()+[RL.epsilon])
            agent.act(action)
            sensor.update(scene,agent)
            # scene.update()
            observation_ *= hp.fading_mem
            observation_ += local_observer(sensor, agent)  # todo: generalize
            RL.store_transition(observation.reshape([-1]), action, reward.reward, observation_.reshape([-1]))
            observation = copy.copy(observation_)
            step += 1
            if (step > 100) and (step % hp.steps_between_learnings == 0):
                RL.learn()
            if step%1000 ==0:
                print(episode,step,' running reward   ',running_ave_reward)
                print('frame:', scene.current_frame,)
                if running_ave_reward[0] > best_thus_far:
                    best_thus_far = running_ave_reward[0]
                    RL.dqn.save_nwk_param(hp.this_run_path+'best_liron.nwk')
                    print('saved best network, mean reward: ', best_thus_far)
            if step%10000 ==0:
                    recorder.plot()
                    RL.dqn.save_nwk_param(hp.this_run_path+'tempX_1.nwk')
                    # debug_policy_plot()
            # if step % 100000 == 0:
            #         recorder.save(hp.this_run_path+recorder_file)
    # recorder.save(hp.this_run_path+recorder_file)


if __name__ == "__main__":

    lsbjob = os.getenv('LSB_JOBID')
    lsbjob = '' if lsbjob is None else lsbjob

    hp = HP()
    hp.save_path = 'saved_runs'

    # hp.description = "only 2nd image from videos 1st frame, penalty for speed, soft q learning"
    hp.description = "new vanilla syclop on mnist"
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
    hp.speed_penalty = 0
    hp.dqn_initial_network = None  # 'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1590422112_0/tempX_1.nwk'  # None #'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1589383850_0/tempX_1.nwk'
    hp.eval_mode = False
    hp.eval_dirs = []
    # hp.eval_dirs =['saved_runs/run_saccader103.py_noname__tau20__1609610126__70900/','saved_runs/run_saccader103.py_noname__hello3__1609401198__31923/']

    parser = argparse.ArgumentParser()
    parser.add_argument('--tau_int', default=4., type=float, help='Integration timescale for adaaptation')
    parser.add_argument('--resize', default=1.0, type=float, help='resize of images')
    parser.add_argument('--run_name_suffix', default='', type=str, help='suffix for runname')
    parser.add_argument('--color', default='grayscale', type=str, help='grayscale/rgb')
    parser.add_argument('--speed_penalty',  default=1.0, type=float, help='resize of images')

    config = parser.parse_args()
    config = vars(config)
    hp.upadte_from_dict(config)
    #finish with updates
    if hp.fisheye_file is None:
        fy_dict = None
    else:
        with open(hp.fisheye_file, 'rb') as f:
            fy_dict = pickle.load(f)
    hp.this_run_name = sys.argv[0] + '_noname_' + hp.run_name_suffix + '_' + str(int(time.time())) + '_' + lsbjob
    hp.grayscale = hp.color == 'grayscale'
    nchannels = 1 if hp.grayscale else 3



    recorder = Recorder(n=6)

    images = prep_mnist_padded_images(5000)

    # with open('../video_datasets/liron_images/shuffled_images.pkl', 'rb') as f:
    #     images = pickle.load(f)

    scene = syc.Scene(frame_list=images)
    sensor = syc.Sensor( log_mode=False, log_floor = 1.0)
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])

    reward = syc.Rewards(reward_types=['central_rms_intensity', 'speed','saccade'],relative_weights=[1.0,hp.speed_penalty,-200])
    # observation_size = sensor.hp.winx*sensor.hp.winy*2
    observation_size = 256*4
    RL = DeepQNetwork(len(agent.hp.action_space), observation_size*hp.mem_depth,#sensor.frame_size+2,
                      reward_decay=0.99,
                      e_greedy=0.95,
                      e_greedy0=0.8,
                      replace_target_iter=10,
                      memory_size=100000,
                      e_greedy_increment=0.0001,
                      learning_rate=0.0025,
                      double_q=True,
                      dqn_mode=True,
                      state_table=np.zeros([1,observation_size*hp.mem_depth]),
                      soft_q_type='boltzmann',
                      beta_schedule=[[400000//hp.steps_between_learnings, 0.1], [700000//hp.steps_between_learnings, 1]],
                      arch='mlp')

    if not(hp.dqn_initial_network is None):
        RL.dqn.load_nwk_param(hp.dqn_initial_network)
    hp.scene = scene.hp
    hp.sensor = sensor.hp
    hp.agent = agent.hp
    hp.reward = reward.hp
    hp.RL = RL.hp
    deploy_logs()
    with open(hp.this_run_path+hp_file, 'wb') as f:
        pickle.dump(hp, f)
    run_env()
    print('results are in:', hp.this_run_path)


