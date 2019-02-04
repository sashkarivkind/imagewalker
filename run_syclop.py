from image_env_mnist1 import Image_env1
from RL_brain_b import DeepQNetwork
import numpy as np
import time
import SYCLOP_env as syc
from misc import *
hp=HP()
hp.max_episode = 3000
hp.steps_per_episode = 10000
hp.steps_between_learnings = 1000


def run_img():
    step = 0
    for episode in range(hp.max_episode):
        observation = env.reset()
        for step in range(hp.steps_per_episode):
            action = RL.choose_action(observation)
            agent.act(action)
            sensor.update(scene,agent)

            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward.reward, observation_)
            if (step > 100) and (step % hp.steps_between_learnings == 0):
                RL.learn()
            if
            env.plot_reward()
            observation = observation_

            step += 1
        env.save_train_history()

if __name__ == "__main__":

    vertical_edge_mat = np.zeros([28,28])
    vertical_edge_mat[:,14:] = 1

    scene = syc.Scene(image_matrix=vertical_edge_mat)
    sensor = syc.Sensor()
    agent = syc.Agent(max_q = [scene.maxx-sensor.hp.winx,scene.maxy-sensor.hp.winy])
    reward = syc.Rewards()

    hp.scene = scene.hp
    hp.sensor = sensor.hp
    hp.agent = agent.hp
    hp.reward = reward.hp

    RL = DeepQNetwork(len(agent.hp.action_space), sensor.frame_size,
                      reward_decay=0.5,
                      e_greedy=0.8,
                      e_greedy0=0.5,
                      replace_target_iter=20,
                      memory_size=300000,
                      e_greedy_increment=0.001,
                      state_table=all_observations_for_mapping
                      )
    run_env()


