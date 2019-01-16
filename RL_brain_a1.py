"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import tensorflow as tf
import scipy.spatial.distance as ssd

np.random.seed(1)
tf.set_random_seed(1)
np.set_printoptions(threshold=np.nan)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            e_greedy0=0.5,
            replace_target_iter=30,
            memory_size=500,
            batch_size=512,
            max_qlearn_steps = 1,
            qlearn_tol = 1e-2,
            e_greedy_increment=None,
            output_graph=False,
            state_table=None
    ):
        self.dqn_mode = False
        self.table_alpha = 0.1
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon0 = e_greedy0
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.max_qlearn_steps = max_qlearn_steps
        self.qlearn_tol = qlearn_tol
        self.epsilon = self.epsilon0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        if not self.dqn_mode:
            self.state_table = state_table
            self.q_eval = np.zeros((np.shape(state_table)[0],n_actions))
        else:
            self._build_net()
            t_params = tf.get_collection('target_net_params')
            e_params = tf.get_collection('eval_net_params')
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            self.sess = tf.Session()

            if output_graph:
                # $ tensorboard --logdir=logs
                # tf.train.SummaryWriter soon be deprecated, use following
                tf.summary.FileWriter("logs/", self.sess.graph)

            self.sess.run(tf.global_variables_initializer())
            self.cost_his = []

    def approx_by_table_entry(self, states):
        return np.argmin(ssd.cdist(self.state_table,states), axis=0)

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        n_l1 = 30
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names,  w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES],  \
                tf.random_normal_initializer(0., 1.0/np.sqrt(n_l1)), tf.constant_initializer(0.001)  # config of layers

            # first layer. collections is used later when assign to target net
            #with tf.variable_scope('debug'):
            #    w0=tf.get_variable('w1', [self.n_features, self.n_actions]], initializer=w_initializer, collections=c_names)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.tanh(tf.matmul(self.s, w1) + b1)

            # hidden layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)


            # out layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            #self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss,var_list=[w3,b3])

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.tanh(tf.matmul(self.s_, w1) + b1)

            # hidden layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)


            # out layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

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

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.compute_q_eval(observation)#self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            # print('optimal action')
        else:
            action = np.random.randint(0, self.n_actions)
            # print('random action')
        # print('observation=', observation, 'action =', action)
        # print('-------------------')
        return action

    def compute_q_eval(self, state):
        if self.dqn_mode:
            return self.sess.run(self.q_eval, feed_dict={self.s: state})
        else:
            ii = np.argmin(np.sum((self.state_table - state)**2,axis=1)) #todo - generalize beyond eucledian distance
            #print(ii, state)
            return self.q_eval[ii,:]

    def map_actions(self, observation): #todo rewrite in matrix form
        actions_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        return actions_values

    def learn(self):
        if self.dqn_mode:
            # check to replace target parameters
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.replace_target_op)
                print('\ntarget_params_replaced\n')

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
            q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={
                    self.s_: batch_memory[:, -self.n_features:],  # fixed network params
                    self.s: batch_memory[:, :self.n_features],  # newest network params are used to compute cur
                })

            # change q_target w.r.t q_eval's action
            q_target = q_eval.copy()

            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            stopflag = False
            qlearn_step = 0
            # train eval network
            print('feed dict', np.concatenate([batch_memory[:, :self.n_features], q_target],axis=1))
            while not stopflag:
                _, self.cost = self.sess.run([self._train_op, self.loss],
                                             feed_dict={self.s: batch_memory[:, :self.n_features],
                                                        self.q_target: q_target})
                qlearn_step +=1
                stopflag  = self.cost < self.qlearn_tol or qlearn_step > self.max_qlearn_steps
                print('cost', self.cost)
            self.cost_his.append(self.cost)
        else:
            batch_table_index = self.approx_by_table_entry(batch_memory[:, :self.n_features])  # np.argmin(np.sqrt((self.state_table -batch_memory[:, :self.n_features] )**2))- #todo generalize the distance
            batch_table_index_ = self.approx_by_table_entry(batch_memory[:, -self.n_features:])  # np.argmin(np.sqrt((self.state_table -batch_memory[:, :self.n_features] )**2))- #todo generalize the distance
            self.q_eval[batch_table_index, eval_act_index] = (1-self.table_alpha)*self.q_eval[batch_table_index, eval_act_index] + \
                self.table_alpha*(reward + self.gamma * np.max(self.q_eval[batch_table_index_, :], axis=1))


        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learn counter', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



