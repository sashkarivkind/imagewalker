"""updating centralview size to 32 and planning to add cifar based decoder"""
#from image_env_mnist1 import Image_env1
from RL_saccader_x1 import DeepQNetwork
from RL_networks import Stand_alone_net
import numpy as np
import time
import pickle
import copy
import SYCLOP_env as syc
from misc import *
import sys
import os
import tensorflow as tf
import cv2
# cv2.ocl.setUseOpenCL(True)


hp=HP()
# if not os.path.exists(hp.this_run_path):
#     os.makedirs(hp.this_run_path)
# else:
#     error('run name already exists!')

lsbjob=os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob

hp.save_path = 'saved_runs'
hp.this_run_name = sys.argv[0] + '_noname_' + sys.argv[-1] + '_' + str(int(time.time())) +'_' + lsbjob

# hp.description = "only 2nd image from videos 1st frame, penalty for speed, soft q learning"
hp.description = "padding+fishyey + drift network co-trained, on Stanford dataset"
hp.mem_depth = 1
hp.padding=[32,32]
hp.max_episode = 10000
hp.steps_per_episode = 100
hp.steps_between_learnings = 100
hp.ae_steps_to_training = 1e10
hp.steps_before_learning_begins = 100
hp.saccade_observation_scale = 100
hp.fading_mem = 0.0
hp.drift_signal_to_saccade_en = 1
hp.fisheye_file = 'fisheye_102.pkl'
hp.tau_int = int(sys.argv[1])
hp.image_path = os.getenv('HOME')+'/datasets/Stanford40/JPEGImages/*'

hp.images_per_scene=20
recorder_file = 'records.pkl'
hp_file = 'hp.pkl'
hp.drift_state_size=200
hp.dqn_initial_network = None # 'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1590422112_0/tempX_1.nwk'  # None #'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1589383850_0/tempX_1.nwk'
hp.drift_initial_network= 'ref_nets/cifar_nwk_102.pkl' #'ref_nets/drift_net_sf40_ae2//trained.nwk' #'ref_nets/drift_net1/trained.nwk'
hp.drift_net_abs_en = False
hp.eval_mode = False
hp.eval_dirs = []

# hp.eval_dirs =['saved_runs/run_saccader103.py_noname__tau20__1609610126__70900/','saved_runs/run_saccader103.py_noname__hello3__1609401198__31923/']
if hp.fisheye_file is None:
    fy_dict = None
else:
    with open(hp.fisheye_file,'rb') as f:
        fy_dict=pickle.load(f)

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

def local_observer(sensor,agent,integrator):
    normfactor=1.0/256.0
    return np.concatenate([normfactor*sensor.dvs_view.reshape([-1]), agent.qdot, integrator.reshape([-1])],axis=0)

def saccade_observer(sensor,drift_observation):
    normfactor=1.0/256.0
    return np.concatenate([normfactor*sensor.frame_view.reshape([-1]), hp.drift_signal_to_saccade_en*drift_observation.reshape([-1])],axis=0)

def drift_state_for_saccade_observation(current_state, integrator_state):
    return integrator_state

def drift_state_for_integrator(drift_net_state, abs_en=False,layer=3):
    if abs_en:
        return np.abs(drift_net_state[layer])
    else:
        return drift_net_state[layer]

def index_to_coord(i,xymax,offset=[0,0]):
    return [i%xymax[1]+offset[0],-(i//xymax[1]+offset[1])]


def run_env(eval_mode=False):
    scene = None
    old_policy_map=0
    step = 0
    best_thus_far = -1e10
    running_ave_reward = 0
    running_timing = 0
    drift_observations = []
    tlist=[0]*13
    reward_and_rewards_list=[]
    for episode in range(hp.max_episode):
        saccade_observation = 0*np.random.uniform(0,1,size=[hp.mem_depth, saccade_observation_size])
        saccade_observation_ = 0*np.random.uniform(0,1,size=[hp.mem_depth, saccade_observation_size])
        image, _ = read_random_image_from_path(hp.image_path, grayscale=True, padding=hp.padding)
        del scene
        # print('---------------------------------------------------')
        scene = syc.Scene(image_matrix=image)
        saccade_agent.reset(
            q_reset=np.array([100, 100]),
            max_q=[scene.maxx - sensor.hp.winx, scene.maxy - sensor.hp.winy])

        integrator_state = 0
        saccade_agent.reset()

        sensor.update(scene, saccade_agent)
        sensor.update(scene, saccade_agent)
        saccade_action = 64 * 32 + 32

        for step_prime in range(hp.steps_per_episode):
            # print('debug saccade_agent.q:', saccade_agent.q)
            tlist[0] = time.time()

            drift_observation = 1. / 256 * sensor.central_frame_view.reshape([1,32,32,1])
            drift_observations.append(drift_observation.squeeze())
            drift_net_state = drift_net.eval_incl_layers(drift_observation)
            # if (not eval_mode) and step % hp.ae_steps_to_training == 0 and len(drift_observations) > 0:
            #     xx = np.array(drift_observations)
            #     _, loss = drift_net.training_step(xx,xx) #train ae
            #     drift_observations=[]
            #     print('ae loss:', loss)

            # todo - change to some integrated version of the DVS view when adding the drift loop
            tlist[1] = time.time()
            high_pass_layer = drift_state_for_integrator(drift_net_state, abs_en=hp.drift_net_abs_en) - integrator_state
            tlist[2] = time.time()
            integrator_state = (1 - 1. / hp.tau_int) * integrator_state + (
                        1. / hp.tau_int) * drift_state_for_integrator(drift_net_state)
            tlist[3] = time.time()
            reward.update_rewards(sensor=sensor, agent=saccade_agent, network=high_pass_layer)
            tlist[4] = time.time()
            saccade_RL.store_transition(saccade_observation.reshape([-1]), saccade_action, reward.reward)
            tlist[5] = time.time()
            saccade_observation = hp.saccade_observation_scale * saccade_observer(sensor,
                                                                                  drift_state_for_saccade_observation(
                                                                                      drift_net_state,
                                                                                      integrator_state))
            tlist[6] = time.time()
            saccade_action, ent = saccade_RL.choose_action(saccade_observation.reshape([-1]),
                                                       discard_black_areas=True,
                                                       black_area=(sensor.frame_view>1e-9))
            reward_and_rewards = np.array(
                [reward.reward] + reward.rewards.tolist())
            running_ave_reward = 0.999 * running_ave_reward + 0.001 * reward_and_rewards
            reward_and_rewards_list.append(reward_and_rewards)
            tlist[7] = time.time()

            dq = index_to_coord(saccade_action,sensor.frame_view.shape,offset=[-31,-31])
            tlist[8] = time.time()

            dq_rescaled = dq if hp.fisheye_file is None else undistort_q_poly(dq,fy_dict['w']).squeeze().astype(np.int64)
            tlist[9] = time.time()
            saccade_agent.act(dq_rescaled)

            tlist[10] = time.time()
            sensor.update(scene, saccade_agent)
            tlist[11] = time.time()
            running_timing = 0.999 * running_timing + 0.001 * np.concatenate((np.diff(np.array(tlist[:-1])),[tlist[-1]]))
            # scene.update()
            # observation_ *= hp.fading_mem
            # observation_ += local_observer(sensor, agent,-integrator_state)  # todo: generalize
            # observation = copy.copy(observation_)
            step += 1
            if (not eval_mode) and (step > hp.steps_before_learning_begins) and (step % hp.steps_between_learnings == 0):
                t0=time.time()
                saccade_RL.learn()
                tlist[12]=time.time()-t0
            if (not eval_mode) and step%1000 ==0:
                print(episode,step,' running reward ',running_ave_reward)
                print('  entropy:', ent)
                if (not eval_mode):
                    print('timing = ', running_timing)
                if (not eval_mode) and running_ave_reward[0] > best_thus_far:
                    best_thus_far = running_ave_reward[0]
                    saccade_RL.dqn.save_nwk_param(hp.this_run_path+'best_liron.nwk')
                    print('saved best network, mean reward: ', best_thus_far)
            if (not eval_mode) and step%10000 ==0:
                    # recorder.plot()
                    saccade_RL.dqn.save_nwk_param(hp.this_run_path+'tempX_saccade.nwk')
                    drift_net.save_nwk_param(hp.this_run_path+'tempX_drift.nwk')
                    # debug_policy_plot()
            # if step % 100000 == 0:
            #         recorder.save(hp.this_run_path+recorder_file)
    # recorder.save(hp.this_run_path+recorder_file)
    print('mean: ', np.mean(reward_and_rewards_list,axis=0))
    print('std dev: ', np.std(reward_and_rewards_list,axis=0))
    print('std error: ', np.std(reward_and_rewards_list,axis=0)/np.sqrt(len(reward_and_rewards_list)))



if __name__ == "__main__":

    recorder = Recorder(n=6)


    sensor = syc.Sensor( fisheye=fy_dict,centralwinx=32,centralwiny=32)
    saccade_agent = syc.Saccadic_Agent()

    reward = syc.Rewards(reward_types=['network'],relative_weights=[100.0])
    # observation_size = sensor.hp.winx*sensor.hp.winy*2
    saccade_observation_size = 64*64+hp.drift_state_size

    # saccade_RL = DeepQNetwork(np.prod(saccade_agent.max_q), saccade_observation_size,
    saccade_RL=DeepQNetwork(64*64, saccade_observation_size,
                      n_features_shaped=list(np.shape(sensor.dvs_view))+[1],
                      shape_fun= None,
                      reward_decay=0.99,
                      replace_target_iter=10,
                      memory_size=100000,
                      e_greedy_increment=0.0001,
                      learning_rate=0.0025,
                      double_q=True,
                      dqn_mode=True,
                      soft_q_type='boltzmann',
                      beta_schedule=[[400000//hp.steps_between_learnings, 1], [700000//hp.steps_between_learnings, 10]],
                      arch='conv_saccades_v1',
                    n_modulating_features=hp.drift_state_size
                      )
    # at this point drift network is a standalone network taken from some external source (e.g. pretrained)
    # in future it will be an action generating network from the drift loop
    # drift_net = Stand_alone_net(16*16,10,arch='mlp', layer_size = [None]+[100]+[100]+[ None])
    drift_net = Stand_alone_net([32,32,1],10,arch='conv',
                                    layer_size = [None]+[[3,3,32]]+[[2,2,16]]+[200]+[ None],
                                    loss_type='softmax_cross_entropy',
                                    trainable=True,
                                    lr=0.0005,
                                    lambda_reg=0.0) #simple cifar10 classifier
    drift_net.assign_session_to_nwk(saccade_RL.dqn.sess)
    saccade_RL.dqn.sess.run(tf.global_variables_initializer())
    saccade_RL.dqn.reset()
    if not(hp.drift_initial_network is None):
        drift_net.load_nwk_param(hp.drift_initial_network)
    if not(hp.dqn_initial_network is None):
        saccade_RL.dqn.load_nwk_param(hp.dqn_initial_network)
    # hp.scene = scene.hp
    print('debug hp',sensor.hp.centralwinx,sensor.hp.centralwiny)
    hp.sensor = sensor.hp
    hp.saccade_agent = saccade_agent.hp
    hp.reward = reward.hp
    hp.saccade_RL = saccade_RL.hp
    deploy_logs()
    with open(hp.this_run_path+hp_file, 'wb') as f:
        pickle.dump(hp, f)
    if not hp.eval_mode:
        run_env()
    else:
        for run_dir in hp.eval_dirs:
            drift_net.load_nwk_param(run_dir+'tempX_drift.nwk')
            saccade_RL.dqn.load_nwk_param(run_dir+'tempX_saccade.nwk')
            hp.update_attributes_from_file(run_dir+'hp.pkl',['tau_int'])
            print('Testing results from:', run_dir)
            print('sanity check, tau_int', hp.tau_int)
            saccade_RL.beta=1e-10
            print('Setting random policy, beta=',saccade_RL.beta)
            run_env(eval_mode=True)
            saccade_RL.beta=10
            print('Setting large beta policy, beta=',saccade_RL.beta)
            run_env(eval_mode=True)

    print('results are in:', hp.this_run_path)


