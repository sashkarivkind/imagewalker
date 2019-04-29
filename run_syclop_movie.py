from image_env_mnist1 import Image_env1
from RL_brain_b import DeepQNetwork
import numpy as np
import time
import pickle
import copy
import SYCLOP_env as syc
from misc import *
hp=HP()
hp.mem_depth = 1
hp.max_episode =  1000
hp.steps_per_episode = 2000
hp.steps_between_learnings = 100
hp.fading_mem = 0.5
recorder_file = 'records.pkl'
hp_file = 'hp.pkl'
hp.contrast_range = [1.0,1.1]
from mnist import MNIST


def local_observer(sensor,agent):
    return np.concatenate([1.0/65000*(sensor.dvs_view.reshape([-1]))])

def run_env():
    old_policy_map=0
    step = 0
    best_thus_far = -1e10
    for episode in range(hp.max_episode):
        observation = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        observation_ = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        agent.reset()
        sensor.reset()
        sensor.update(scene, agent)
        sensor.update(scene, agent)

        for step_prime in range(hp.steps_per_episode):
            action = RL.choose_action(observation.reshape([-1]))
            reward.update_rewards(sensor = sensor, agent = agent)
            recorder.record([agent.q_ana[0],agent.q_ana[1],reward.reward,RL.epsilon])
            agent.act(action)
            sensor.update(scene,agent)
            scene.update()
            observation_ *= hp.fading_mem
            observation_ += local_observer(sensor, agent)  # todo: generalize
            RL.store_transition(observation.reshape([-1]), action, reward.reward, observation_.reshape([-1]))
            observation = copy.copy(observation_)
            step += 1
            if (step > 100) and (step % hp.steps_between_learnings == 0):
                RL.learn()
            if step%1000 ==0:
                print(episode,step)
                print('frame:', scene.current_frame)
                if recorder.running_averages[2][-1] > best_thus_far:
                    best_thus_far = recorder.running_averages[2][-1]
                    RL.dqn.save_nwk_param('best_movie.nwk')
                    print('saved best network, mean reward: ', best_thus_far)
            if step%10000 ==0:
                    recorder.plot()
                    RL.dqn.save_nwk_param('tempX_1.nwk')
                    # debug_policy_plot()
            if step % 100000 == 0:
                    recorder.save(recorder_file)
    recorder.save(recorder_file)


if __name__ == "__main__":

    recorder = Recorder(n=4)

    scene = syc.Scene(frame_list=read_images_from_path(path='/home/bnapp/arivkindNet/video_datasets/dataset-corridor1_512_16/mav0/cam0/data/*.png'))
    sensor = syc.Sensor( log_mode=False, log_floor = 1.0)
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])

    reward = syc.Rewards()
    observation_size = sensor.hp.winx*sensor.hp.winy
    RL = DeepQNetwork(len(agent.hp.action_space), observation_size*hp.mem_depth,#sensor.frame_size+2,
                      reward_decay=0.99,
                      e_greedy=0.9,
                      e_greedy0=0.1,
                      replace_target_iter=10,
                      memory_size=100000,
                      e_greedy_increment=0.0001,
                      learning_rate=0.0025,
                      double_q=False,
                      dqn_mode=True,
                      state_table=np.zeros([1,observation_size*hp.mem_depth])
                      )


    hp.scene = scene.hp
    hp.sensor = sensor.hp
    hp.agent = agent.hp
    hp.reward = reward.hp
    hp.RL = RL.hp
    with open(hp_file, 'wb') as f:
        pickle.dump(hp, f)
    run_env()


