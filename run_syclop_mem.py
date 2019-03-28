from image_env_mnist1 import Image_env1
from RL_brain_b import DeepQNetwork
import numpy as np
import time
import SYCLOP_env as syc
from misc import *
hp=HP()
hp.mem_depth = 2
hp.max_episode = 30000
hp.steps_per_episode = 1000
hp.steps_between_learnings = 100

def local_observer(sensor,agent):
    return np.concatenate([np.abs(sensor.dvs_view[5,:].reshape([-1])),1*agent.qdot,0.1*agent.q])

def run_env():
    old_policy_map=0
    step = 0
    best_thus_far = -1e10
    for episode in range(hp.max_episode):
        observation = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        observation_ = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy]) #todo: reset method for agent

        for step_prime in range(hp.steps_per_episode):
            action = RL.choose_action(observation.reshape([-1]))
            reward.update_rewards(sensor = sensor, agent = agent)
            agent.act(action)
            sensor.update(scene,agent)
            observation_[1:,:] = observation_[:-1,:]
            observation_[0,:]  =  local_observer(sensor, agent)  # todo: generalize
            RL.store_transition(observation.reshape([-1]), action, reward.reward, observation_.reshape([-1]))
            recorder.record([agent.q_ana[0],np.argmax(observation[0,:-2]),agent.qdot[0],reward.rewards[0],reward.rewards[1],reward.reward,RL.epsilon])
            if (step > 100) and (step % hp.steps_between_learnings == 0):
                RL.learn()
            observation = observation_
            step += 1
            if step%1000 ==0:
                print(episode,step)
                # policy_map = np.array([np.argmax(RL.compute_q_eval(np.eye(10)), axis=1),
                #                         np.argmax(RL.compute_q_eval(debu2el), axis=1)])
                # if np.max(np.abs(old_policy_map - policy_map))>0:
                #     print(policy_map)
                #     print('policy_change')
                #     print(policy_map-old_policy_map)
                #     print('--------------------------------')
                if recorder.running_averages[3][-1] > best_thus_far:
                    best_thus_far = recorder.running_averages[3][-1]
                    RL.dqn.save_nwk_param('best_thus_farAdam.nwk')
                    print('saved best network, mean reward: ', best_thus_far)
                #     old_policy_map = policy_map
            if step%10000 ==0:
                    recorder.plot()
                    # print('--------state table ----', RL.state_table.shape )
                    RL.dqn.save_nwk_param('tempX_1.nwk')



if __name__ == "__main__":

    vertical_edge_mat = np.zeros([28,128])
    vertical_edge_mat[:,64:] = 1.0
    recorder = Recorder(n=7)
    #debu2el = np.diag(np.ones([10-1]),k=1)+np.eye(10)
    # debu2el = debu2el[:-1,:]

    scene = syc.Scene(image_matrix=vertical_edge_mat)
    sensor = syc.Sensor()
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])
    reward = syc.Rewards()
    observation_size = sensor.hp.winx + 2+2
    RL = DeepQNetwork(len(agent.hp.action_space), observation_size*hp.mem_depth,#sensor.frame_size+2,
                      reward_decay=0.90,
                      e_greedy=0.95,
                      e_greedy0=0.2,
                      replace_target_iter=100,
                      memory_size=30000,
                      e_greedy_increment=0.0005,
                      double_q=True,
                      dqn_mode=True,
                      state_table=np.zeros([1,observation_size*hp.mem_depth])
                      )


    hp.scene = scene.hp
    hp.sensor = sensor.hp
    hp.agent = agent.hp
    hp.reward = reward.hp
    hp.RL = RL.hp

    run_env()


