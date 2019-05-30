from image_env_mnist1 import Image_env1
from RL_brain_b import DeepQNetwork
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
hp.save_path = 'saved_runs'
hp.this_run_name = sys.argv[0] + '_noname_' + str(int(time.time()))
hp.description = "running some stills from something something database, statring from  RL.dqn.save_nwk_param('liron_encircle.nwk')"
hp.mem_depth = 1
hp.max_episode =  10000
hp.steps_per_episode = 100
hp.steps_between_learnings = 100
hp.fading_mem = 0.5
recorder_file = 'records.pkl'
hp_file = 'hp.pkl'
hp.contrast_range = [1.0,1.1]
hp.logmode = False
from mnist import MNIST



if not os.path.exists(hp.save_path):
    os.makedirs(hp.save_path)

hp.this_run_path = hp.save_path+'/'+hp.this_run_name+'/'
if not os.path.exists(hp.this_run_path):
    os.makedirs(hp.this_run_path)
else:
    error('run name already exists!')

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
            action = RL.choose_action(observation.reshape([-1]))
            reward.update_rewards(sensor = sensor, agent = agent)
            running_ave_reward = 0.999*running_ave_reward+0.001*reward.reward
            if step % 10000 < 1000:
                recorder.record([agent.q_ana[0],agent.q_ana[1],reward.reward,RL.epsilon])
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
                if running_ave_reward > best_thus_far:
                    best_thus_far = running_ave_reward
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

    recorder = Recorder(n=4)

    images = read_images_from_path('/home/bnapp/arivkindNet/video_datasets/stills_from_videos/some100img_from20bn/*')

    # images = [np.sum(1.0*uu, axis=2) for uu in images]
    # images = [cv2.resize(uu, dsize=(256, 256-64), interpolation=cv2.INTER_AREA) for uu in images]
    if hp.logmode:
        images = [np.log10(uu+1.0) for uu in images]


    # with open('../video_datasets/liron_images/shuffled_images.pkl', 'rb') as f:
    #     images = pickle.load(f)

    scene = syc.Scene(frame_list=images)
    sensor = syc.Sensor( log_mode=False, log_floor = 1.0)
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])

    reward = syc.Rewards()
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
                      state_table=np.zeros([1,observation_size*hp.mem_depth])
                      )
    # RL.dqn.load_nwk_param('tempX_1.nwk')
    RL.dqn.save_nwk_param('liron_encircle.nwk')
    hp.scene = scene.hp
    hp.sensor = sensor.hp
    hp.agent = agent.hp
    hp.reward = reward.hp
    hp.RL = RL.hp
    with open(hp.this_run_path+hp_file, 'wb') as f:
        pickle.dump(hp, f)
    run_env()
    print('results are in:', hp.this_run_path)


