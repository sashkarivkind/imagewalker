from image_env1a import Image_env1
from RL_brain_a1 import DeepQNetwork
import numpy as np
import time

def run_img():
    step = 0
    for episode in range(3000):
        # initial observation
        observation = env.reset()

        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            #print('sample q entry',observation, action, reward, observation_)
            if (step > 100) and (step % 10000 == 0):
                RL.learn()
                act_map=RL.map_actions(all_observations_for_mapping) if RL.dqn_mode else RL.q_eval
                #print(np.argmax(act_map,axis=1).reshape((env.image_x,env.image_y)))
                print(np.flip(np.array(env.num2srt_actions(np.argmax(act_map, axis=1))).reshape((env.image_x,env.image_y)).transpose(), axis=0))
                print('epsilon', RL.epsilon)
                env.q_snapshots.append([step,all_observations_for_mapping,act_map])
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print(reward)
                print(observation_)
                break
            step += 1

        # time.sleep(2.0)

    # end of game
    print('game over')
    #env.destroy()


if __name__ == "__main__":
    # maze game
    env = Image_env1()
    all_observations_for_mapping = env.observation_space()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.00005,
                      reward_decay=0.9,
                      e_greedy=0.95,
                      e_greedy0=0.8,
                      replace_target_iter=2000,
                      memory_size=1000000,
                      e_greedy_increment=0.001,
                      # output_graph=True
                      state_table=all_observations_for_mapping
                      )
    run_img()
    # env.after(100, run_img)
    # print('-----------------------------------------')
    # env.mainloop()
    env.plot_reward()
    env.save_train_history()
    #RL.plot_cost()
