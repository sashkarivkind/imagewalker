
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

hp=HP()
# if not os.path.exists(hp.this_run_path):
#     os.makedirs(hp.this_run_path)
# else:
#     error('run name already exists!')

hp.save_path = 'saved_runs'
hp.this_run_name = sys.argv[0] + '_noname_' + str(int(time.time()))
# hp.description = "only 2nd image from videos 1st frame, penalty for speed, soft q learning"
# hp.description = "drift network pre-trained on mnist patches, try3-No signal to saccadic net from drift net"
hp.description = "drift network pre-trained on mnist patches AFTER BUG FIX  -No signal to saccadic net from drift net"
hp.mem_depth = 1
hp.max_episode = 10000
hp.steps_per_episode = 100
hp.steps_between_learnings = 100
hp.steps_before_learning_begins = 100
hp.saccade_observation_scale = 100
hp.fading_mem = 0.0
hp.drift_signal_to_saccade_en = 0
hp.tau_int=4
hp.num_images = 200
hp.images_per_scene=20
recorder_file = 'records.pkl'
hp_file = 'hp.pkl'
hp.contrast_range = [1.0,1.1]
hp.dqn_initial_network = None # 'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1590422112_0/tempX_1.nwk'  # None #'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1589383850_0/tempX_1.nwk'
hp.drift_initial_network='ref_nets/drift_net1/trained.nwk'
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
    print('hyper-parameters (partial):', hp)

def local_observer(sensor,agent,integrator):
    normfactor=1.0/256.0
    return np.concatenate([normfactor*sensor.dvs_view.reshape([-1]), agent.qdot, integrator.reshape([-1])],axis=0)

def saccade_observer(sensor,drift_observation):
    normfactor=1.0/256.0
    return np.concatenate([normfactor*sensor.frame_view.reshape([-1]), hp.drift_signal_to_saccade_en*drift_observation.reshape([-1])],axis=0)

def drift_state_for_saccade_observation(current_state, integrator_state):
    return integrator_state

def drift_state_for_integrator(drift_net_state):
    return drift_net_state[2]

def index_to_coord(i,xymax,offset=[0,0]):
    return [i%xymax[1]+offset[0],-(i//xymax[1]+offset[1])]


def run_env():
    old_policy_map=0
    step = 0
    best_thus_far = -1e10
    running_ave_reward = 0

    for episode in range(hp.max_episode):
        saccade_observation = 0*np.random.uniform(0,1,size=[hp.mem_depth, saccade_observation_size])
        saccade_observation_ = 0*np.random.uniform(0,1,size=[hp.mem_depth, saccade_observation_size])
        scene.current_frame = np.random.choice(scene.total_frames)
        scene.image = scene.frame_list[scene.current_frame]
        integrator_state = 0
        saccade_agent.reset()

        sensor.reset()
        sensor.update(scene, saccade_agent)
        sensor.update(scene, saccade_agent)
        saccade_action = 64 * 32 + 32

        for step_prime in range(hp.steps_per_episode):
            # print('debug drift_state:',  sensor.central_frame_view.reshape([1,-1]))
            drift_net_state = drift_net.eval_incl_layers(1. / 256 * sensor.central_frame_view.reshape(
                [1, -1]))  # todo - change to some integrated version of the DVS view when adding the drift loop
            high_pass_layer = drift_state_for_integrator(drift_net_state) - integrator_state
            integrator_state = (1 - 1. / hp.tau_int) * integrator_state + (
                        1. / hp.tau_int) * drift_state_for_integrator(drift_net_state)
            reward.update_rewards(sensor=sensor, agent=saccade_agent, network=high_pass_layer)
            saccade_RL.store_transition(saccade_observation.reshape([-1]), saccade_action, reward.reward)
            saccade_observation = hp.saccade_observation_scale * saccade_observer(sensor,
                                                                                  drift_state_for_saccade_observation(
                                                                                      drift_net_state,
                                                                                      integrator_state))
            saccade_action, ent = saccade_RL.choose_action(saccade_observation.reshape([-1]))
            running_ave_reward = 0.999 * running_ave_reward + 0.001 * np.array(
                [reward.reward] + reward.rewards.tolist())
            # print('testing printout: reward:',reward.reward,' coord' ,saccade_agent.q,'\n')
            # saccade_agent.act(index_to_coord(saccade_action, saccade_agent.max_q, offset=[-31, -31]))
            saccade_agent.act(index_to_coord(saccade_action, sensor.frame_view.shape, offset=[-31, -31]))

            sensor.update(scene, saccade_agent)
            # scene.update()
            # observation_ *= hp.fading_mem
            # observation_ += local_observer(sensor, agent,-integrator_state)  # todo: generalize
            # observation = copy.copy(observation_)
            step += 1
            if (step > hp.steps_before_learning_begins) and (step % hp.steps_between_learnings == 0):
                saccade_RL.learn()
            if step % 10000 < 1000:
                pass
                # recorder.record(
                #     [saccade_agent.q[0], saccade_agent.q[1], reward.reward] + reward.rewards.tolist() + [saccade_RL.epsilon])
            # [agent.q_ana[0], agent.q_ana[1], reward.reward] + reward.rewards.tolist() + [RL.epsilon])
            if step%1000 ==0:
                print(episode,step,' running reward ',running_ave_reward)
                print('frame:', scene.current_frame,'  entropy:', ent)
                if running_ave_reward[0] > best_thus_far:
                    best_thus_far = running_ave_reward[0]
                    saccade_RL.dqn.save_nwk_param(hp.this_run_path+'best_liron.nwk')
                    print('saved best network, mean reward: ', best_thus_far)
            if step%10000 ==0:
                    recorder.plot()
                    saccade_RL.dqn.save_nwk_param(hp.this_run_path+'tempX_saccade.nwk')
                    drift_net.save_nwk_param(hp.this_run_path+'tempX_drift.nwk')
                    # debug_policy_plot()
            if step % 100000 == 0:
                    recorder.save(hp.this_run_path+recorder_file)
    # recorder.save(hp.this_run_path+recorder_file)


if __name__ == "__main__":

    recorder = Recorder(n=6)

    # images = read_images_from_path('/home/bnapp/arivkindNet/video_datasets/stills_from_videos/some100img_from20bn/*',max_image=hp.num_images)
    # images = read_images_from_path('/home/labs/ahissarlab/arivkind/video_datasets/stills_from_videos/some100img_from20bn/*',max_image=hp.num_images)
    images = prep_mnist_sparse_images(hp.num_images,images_per_scene=hp.images_per_scene)


    # with open('../video_datasets/liron_images/shuffled_images.pkl', 'rb') as f:
    #     images = pickle.load(f)
    print(len(images))
    scene = syc.Scene(frame_list=images)
    sensor = syc.Sensor( log_mode=False, log_floor = 1.0)
    saccade_agent = syc.Saccadic_Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])

    reward = syc.Rewards(reward_types=['network'],relative_weights=[100.0])
    # observation_size = sensor.hp.winx*sensor.hp.winy*2
    saccade_observation_size = 64*64+100
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
                      beta_schedule=[[4000, 1], [7000, 10]],
                      arch='conv_saccades_v1'
                      )
    # at this point drift network is a standalone network taken from some external source (e.g. pretrained)
    # in future it will be an action generating network from the drift loop
    drift_net = Stand_alone_net(16*16,10,arch='mlp', layer_size = [None]+[100]+[100]+[ None])
    drift_net.assign_session_to_nwk(saccade_RL.dqn.sess)
    saccade_RL.dqn.sess.run(tf.global_variables_initializer())
    saccade_RL.dqn.reset()
    drift_net.load_nwk_param(hp.drift_initial_network)
    if not(hp.dqn_initial_network is None):
        saccade_RL.dqn.load_nwk_param(hp.dqn_initial_network)
    hp.scene = scene.hp
    hp.sensor = sensor.hp
    hp.saccade_agent = saccade_agent.hp
    hp.reward = reward.hp
    hp.saccade_RL = saccade_RL.hp
    deploy_logs()
    with open(hp.this_run_path+hp_file, 'wb') as f:
        pickle.dump(hp, f)
    run_env()
    print('results are in:', hp.this_run_path)


