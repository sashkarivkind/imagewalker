"""
Was developed based on: https://morvanzhou.github.io/tutorials/

"""

import numpy as np
import tensorflow as tf
import scipy.spatial.distance as ssd
import RL_networks as rlnet
from misc import *
np.random.seed(1)
tf.set_random_seed(1)
# np.set_printoptions(threshold=np.nan)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.00025,
            reward_decay=0.9,
            e_greedy=0.9,
            e_greedy0=0.5,
            replace_target_iter=30,
            memory_size=3000,
            batch_size=512,
            max_qlearn_steps = 1,
            qlearn_tol = 1e-2,
            double_q = False,
            e_greedy_increment=None,
            output_graph=False,
            state_table=None,
            table_alpha = 1.0,
            dqn_mode=True,
            soft_q_type=None, #'boltzmann'
            beta=1
            ):

        self.hp = HP()
        #todo: a more elegant integration between hyper parameters and other attributes of RL.
        self.hp.dqn_mode = dqn_mode
        self.hp.table_alpha = table_alpha
        self.hp.n_actions = n_actions
        self.hp.n_features = n_features
        self.hp.lr = learning_rate
        self.hp.gamma = reward_decay
        self.hp.epsilon_max = e_greedy
        self.hp.epsilon0 = e_greedy0
        self.hp.replace_target_iter = replace_target_iter
        self.hp.memory_size = memory_size
        self.hp.batch_size = batch_size
        self.hp.double_q = double_q
        self.hp.epsilon_increment = e_greedy_increment
        self.hp.max_qlearn_steps = max_qlearn_steps
        self.hp.qlearn_tol = qlearn_tol
        self.hp.soft_q_type = soft_q_type
        self.hp.beta = beta
        #todo:avoid redundancy here currently self.hp. is communicated to the upper level, but what is actually used in the code is self. w/o hp!
        self.dqn_mode = dqn_mode
        self.table_alpha = table_alpha
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon0 = e_greedy0
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.double_q = double_q
        self.epsilon_increment = e_greedy_increment
        self.max_qlearn_steps = max_qlearn_steps
        self.qlearn_tol = qlearn_tol
        self.epsilon = self.epsilon0 if e_greedy_increment is not None else self.epsilon_max
        self.soft_q_type = soft_q_type
        self.beta = beta


# total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        if not self.dqn_mode:
            self.state_table = state_table
            self.q_eval = np.zeros((np.shape(state_table)[0],n_actions))
        else:
            self.cost_his = []
            self.dqn = rlnet.DQN_net(n_features, n_actions, learning_rate=self.lr)
            self.dqn.sess = tf.Session()
            self.dqn.sess.run(tf.global_variables_initializer())
            self.dqn.reset()


    def approx_by_table_entry(self, states):
        return np.argmin(ssd.cdist(self.state_table,states), axis=0)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        actions_value = self.compute_q_eval(observation) #todo - this computation was taken out of if to ensure all states are added

        # self.current_val=np.max(actions_value) #todo debud
        # self.delta_val=np.max(actions_value)-np.min(actions_value) #todo debud

        if self.soft_q_type == 'boltzmann':
            boltzmann_measure = np.exp(self.beta * (actions_value-np.max(actions_value))) #todo here substracted max to avoid exponent exploding. need to be taken into a separate function!
            boltzmann_measure = boltzmann_measure / np.sum(boltzmann_measure, axis=1)
            ppp=np.abs(np.sum(boltzmann_measure)-1)
            if ppp>1e-5:
                print('debug prob:',ppp,'------', actions_value,'-----------',boltzmann_measure)
            action = np.random.choice(list(range(self.n_actions)),1, p=boltzmann_measure.reshape([-1]))[0]
        else:
            if np.random.uniform() < self.epsilon:
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
        return action

    def compute_q_eval(self, state, match_th=1e-5):
        if self.dqn_mode:
            return self.dqn.eval(state)
        else:
            dd=(np.sum((self.state_table - state)**2,axis=1)) #todo - generalize beyond eucledian distance
            if np.min(dd) < match_th:
                ii = np.argmin(dd)
            else:
                self.state_table = np.vstack([self.state_table, state])
                self.q_eval =np.vstack([self.q_eval,np.zeros([self.n_actions])])
                ii = self.state_table.shape[0] - 1
            #print(ii, state)
            return self.q_eval[ii,:]

    def map_actions(self, observation): #todo rewrite in matrix form
        actions_values = self.dqn.eval(observation)
        return actions_values

    def learn(self):
        if self.dqn_mode:
            # check to replace target parameters
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.dqn.update_q_next()
                # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int) #after the features comes the action element
        reward = batch_memory[:, self.n_features + 1]

        if self.dqn_mode:
            q_next, q_eval = [self.dqn.eval_next(batch_memory[:, -self.n_features:]),
                             self.dqn.eval( batch_memory[:, :self.n_features])]

            # change q_target w.r.t q_eval's action
            q_target = q_eval.copy()
            self.debu1 = q_target.copy()
            # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            if self.double_q:
                q_next_estim=max_by_different_argmax(q_next,q_eval,axis=1)
            else:
                q_next_estim=np.max(q_next, axis=1)

            if self.soft_q_type=='uniform': #todo combine with the previous if
                q_next_estim = (1-self.soft_q_factor)*q_next_estim+self.soft_q_factor*np.mean(q_next, axis=1)
            elif self.soft_q_type=='boltzmann':
                boltzmann_measure = np.exp(self.beta*(q_next-np.max(q_next,axis=1).reshape([-1,1])))
                boltzmann_measure = boltzmann_measure/np.sum(boltzmann_measure,axis=1).reshape([-1,1])
                q_next_estim=np.sum(boltzmann_measure*q_next, axis=1)
            elif self.soft_q_type is None:
                pass
            else:
                error('unsupoorted type of soft q')
            q_target[batch_index, eval_act_index] = (1-self.table_alpha)*q_target[batch_index, eval_act_index] + self.table_alpha*(reward + self.gamma * q_next_estim)
            self.debu2 = q_target.copy()
            stopflag = False
            qlearn_step = 0
            # train eval network
            #print('feed dict', np.concatenate([batch_memory[:, :self.n_features], q_target],axis=1))
            while not stopflag:
                # print(batch_memory[:, :self.n_features])
                # print('---------------------qqq:')
                _, self.cost = self.dqn.training_step(batch_memory[:, :self.n_features],q_target)

                qlearn_step +=1
                stopflag  = self.cost < self.qlearn_tol or qlearn_step > self.max_qlearn_steps
                #print('cost', self.cost)
            self.cost_his.append(self.cost)
        else:
            batch_table_index = self.approx_by_table_entry(batch_memory[:, :self.n_features])  # np.argmin(np.sqrt((self.state_table -batch_memory[:, :self.n_features] )**2))- #todo generalize the distance
            batch_table_index_ = self.approx_by_table_entry(batch_memory[:, -self.n_features:])  # np.argmin(np.sqrt((self.state_table -batch_memory[:, :self.n_features] )**2))- #todo generalize the distance
            self.q_eval[batch_table_index, eval_act_index] = (1-self.table_alpha)*self.q_eval[batch_table_index, eval_act_index] + \
                self.table_alpha*(reward + self.gamma * np.max(self.q_eval[batch_table_index_, :], axis=1))


        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        #print('learn counter', self.learn_step_counter, 'epsilon: ', self.epsilon)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



