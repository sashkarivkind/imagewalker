from image_env_mnist1 import Image_env1
from RL_brain_b import DeepQNetwork
import numpy as np
import time
import pickle
import copy
import SYCLOP_env as syc
from misc import *
hp=HP()
hp.mem_depth = 3
hp.max_episode = 30000
hp.steps_per_episode = 100
hp.steps_between_learnings = 100

recorder_file = 'records.pkl'
hp_file = 'hp.pkl'

def debug_policy_plot():
    qq_right = []
    x_vec = np.arange(vagent.max_q[0])
    for xx in x_vec:
            vagent.q[0] = xx
            vsensor.update(vscene,vagent)
            vobservation  = local_observer(vsensor,vagent) #todo: generalize
            qq_right.append(RL.compute_q_eval(vobservation.reshape([1,-1])))
    qq_right = (np.reshape(qq_right,[-1,2]))
    vax[0].clear()
    vax[0].plot(qq_right, 'x-')
    vax[1].clear()
    vax[1].plot(qq_right[:,1]-qq_right[:,0], 'x-')

def local_observer(sensor,agent):
    return np.concatenate([10*np.abs(sensor.dvs_view[5,:].reshape([-1])),0*agent.qdot,0.0*agent.q])

def run_env():
    old_policy_map=0
    step = 0
    best_thus_far = -1e10
    for episode in range(hp.max_episode):
        observation = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        observation_ = np.random.uniform(0,1,size=[hp.mem_depth, observation_size])
        # agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy]) #todo: reset method for agent
        agent.reset()
        sensor.reset()
        sensor.update(scene, agent)
        sensor.update(scene, agent)

        for step_prime in range(hp.steps_per_episode):
            action = RL.choose_action(observation.reshape([-1]))
            reward.update_rewards(sensor = sensor, agent = agent)
            recorder.record([agent.q_ana[0],RL.current_val,RL.delta_val,reward.rewards[0],reward.rewards[1],reward.reward,RL.epsilon])
            agent.act(action)
            sensor.update(scene,agent)
            observation_[1:,:] = observation_[:-1,:]
            observation_[0,:]  =  local_observer(sensor, agent)  # todo: generalize
            RL.store_transition(observation.reshape([-1]), action, reward.reward, observation_.reshape([-1]))
            # print('debug0', observation.reshape([-1]))
            # print('debug1', observation_.reshape([-1]))
            # print('debug2', np.max( observation.reshape([-1])-observation_.reshape([-1])),'action:',action,'pos: ',np.argmax(observation.reshape([-1])))
            observation = copy.copy(observation_)
            step += 1
            if (step > 100) and (step % hp.steps_between_learnings == 0):
                RL.learn()
            if step%1000 ==0:
                print(episode,step)
                if recorder.running_averages[3][-1] > best_thus_far:
                    best_thus_far = recorder.running_averages[3][-1]
                    RL.dqn.save_nwk_param('best_ttt.nwk')
                    print('saved best network, mean reward: ', best_thus_far)
            if step%10000 ==0:
                    recorder.plot()
                    RL.dqn.save_nwk_param('tempX_1.nwk')
                    # debug_policy_plot()
            if step % 100000 == 0:
                    recorder.save(recorder_file)
    recorder.save(recorder_file)


if __name__ == "__main__":

    vertical_edge_mat = np.zeros([28,128])
    vertical_edge_mat[:,64:] = 1.0
    recorder = Recorder(n=7)
    #debu2el = np.diag(np.ones([10-1]),k=1)+np.eye(10)
    # debu2el = debu2el[:-1,:]

    scene = syc.Scene(image_matrix=vertical_edge_mat)
    sensor = syc.Sensor()
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])

    vfig, vax = plt.subplots(2,1)

    vscene = syc.Scene(image_matrix=vertical_edge_mat)
    vsensor = syc.Sensor()
    vagent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])

    reward = syc.Rewards()
    observation_size = sensor.hp.winx + 2+2
    RL = DeepQNetwork(len(agent.hp.action_space), observation_size*hp.mem_depth,#sensor.frame_size+2,
                      reward_decay=0.99,
                      e_greedy=0.9,
                      e_greedy0=0.8,
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


