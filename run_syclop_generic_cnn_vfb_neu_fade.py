
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

hp=HP()
# if not os.path.exists(hp.this_run_path):
#     os.makedirs(hp.this_run_path)
# else:
#     error('run name already exists!')

hp.save_path = 'saved_runs'
hp.this_run_name = sys.argv[0] + '_noname_' + str(int(time.time()))
# hp.description = "only 2nd image from videos 1st frame, penalty for speed, soft q learning"
hp.description = "cnn on stills from movies, changed tau_int"
hp.mem_depth = 1
hp.max_episode = 1000
hp.steps_per_episode = 1000
hp.steps_between_learnings = 100
hp.steps_before_learning_begins = 10000
hp.fading_mem = 0.0
hp.tau_int=1e2

recorder_file = 'records.pkl'
hp_file = 'hp.pkl'
hp.contrast_range = [1.0,1.1]
hp.logmode = False
hp.dqn_initial_network = None # 'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1590422112_0/tempX_1.nwk'  # None #'saved_runs/run_syclop_generic_cnn_vfb_neu.py_noname_1589383850_0/tempX_1.nwk'
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

def run_env():
    old_policy_map=0
    step = 0
    best_thus_far = -1e10
    running_ave_reward = 0

    for episode in range(hp.max_episode):
        observation = 0*np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        observation_ = 0*np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        scene.current_frame = np.random.choice(scene.total_frames)
        scene.image = scene.frame_list[scene.current_frame]
        integrator_layer = 0
        agent.reset()

        sensor.reset()
        sensor.update(scene, agent)
        sensor.update(scene, agent)
        for step_prime in range(hp.steps_per_episode):
            net =  RL.dqn.eval_incl_layers(RL.shape_fun(observation.reshape([1, -1])))
            integrator_layer = (1-1./hp.tau_int)*integrator_layer+(1./hp.tau_int)*net[2]
            action, ent = RL.choose_action(observation.reshape([-1]))
            reward.update_rewards(sensor = sensor, agent = agent, network=net[2]-integrator_layer)
            running_ave_reward = 0.999*running_ave_reward+0.001*np.array([reward.reward]+reward.rewards.tolist())
            if step % 10000 < 1000:
                # print([agent.q_ana[0], agent.q_ana[1], reward.reward] , reward.rewards , [RL.epsilon])
                # print(type([agent.q_ana[0], agent.q_ana[1], reward.reward]) , type(reward.rewards), type([RL.epsilon]))
                recorder.record([agent.q_ana[0],agent.q_ana[1],reward.reward]+reward.rewards.tolist()+[RL.epsilon])
            agent.act(action)
            sensor.update(scene,agent)
            # scene.update()
            observation_ *= hp.fading_mem
            observation_ += local_observer(sensor, agent,-integrator_layer)  # todo: generalize
            RL.store_transition(observation.reshape([-1]), action, reward.reward, observation_.reshape([-1]))
            observation = copy.copy(observation_)
            step += 1
            if (step > hp.steps_before_learning_begins) and (step % hp.steps_between_learnings == 0):
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
            if step % 100000 == 0:
                    recorder.save(hp.this_run_path+recorder_file)
    recorder.save(hp.this_run_path+recorder_file)


if __name__ == "__main__":

    recorder = Recorder(n=6)

    # images = read_images_from_path('/home/bnapp/arivkindNet/video_datasets/stills_from_videos/some100img_from20bn/*')
    # images = some_resized_mnist(n=400)
    # images = prep_mnist_padded_images(5000)
    # for ii,image in enumerate(images):
    #     if ii%2:
    #         images[ii]=-image+np.max(image)
    # images = prep_mnist_sparse_images(400,images_per_scene=20)
    images = read_images_from_path('/home/bnapp/arivkindNet/video_datasets/stills_from_videos/some100img_from20bn/*',max_image=10)
    # images = [images[1]]
    # images = [np.sum(1.0*uu, axis=2) for uu in images]
    # images = [cv2.resize(uu, dsize=(256, 256-64), interpolation=cv2.INTER_AREA) for uu in images]
    if hp.logmode:
        images = [np.log10(uu+1.0) for uu in images]


    # with open('../video_datasets/liron_images/shuffled_images.pkl', 'rb') as f:
    #     images = pickle.load(f)

    scene = syc.Scene(frame_list=images)
    sensor = syc.Sensor( log_mode=False, log_floor = 1.0)
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])

    # reward = syc.Rewards(reward_types=['central_rms_intensity', 'speed','network','network_L1'],relative_weights=[1.0,-float(sys.argv[1]),1,-1])
    # reward = syc.Rewards(reward_types=['central_rms_intensity', 'speed','network','network_L1'],relative_weights=[1.0,-float(sys.argv[1]),100,-100*np.sqrt(np.pi/2.)])
    reward = syc.Rewards(reward_types=['central_rms_intensity', 'speed', 'network'],relative_weights=[1.0,-float(sys.argv[1]),0.0])
    # observation_size = sensor.hp.winx*sensor.hp.winy*2
    observation_size = 64*64+\
                       2+\
                       16*16*16 #sensor+velocity+interrmediate layer
    RL = DeepQNetwork(len(agent.hp.action_space), observation_size,
                      n_features_shaped=list(np.shape(sensor.dvs_view))+[1],
                      shape_fun= None,
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
                      beta=0.1,
                      arch='conv_ctrl_fade_v1',
                      train_starting_from_layer=None
                      )
    # RL.dqn.load_nwk_param('tempX_1.nwk')
    # RL.dqn.save_nwk_param('liron_encircle.nwk')
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


