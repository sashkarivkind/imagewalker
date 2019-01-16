import numpy as np
import tensorflow as tf

class DQN_net():
    def __init__(self):
        self.q_eval = Network(trainable=True)
        self.q_next = Network()
        self.sess = None

    def reset(self):
        self.q_next.assign_param_prep(self.q_eval)
        self.update_q_next()

    def assign_session_to_nwk(self, sess):
        self.sess = sess
        self.q_next.sess = sess
        self.q_eval.sess = sess

    def update_q_next(self):
        self.q_next.update(self.sess)

    def training_step(self, observations, q_target):
        return self.sess.run([self.q_eval.train_op,self.q_eval.loss],
                  feed_dict={self.q_eval.observations: observations,
                             self.q_eval.q_target: q_target})

    def eval(self,observations):
        return self.sess.run(self.q_eval.estimator,
                  feed_dict={self.q_eval.observations: observations})

    def eval_next(self,observations):
        return  self.sess.run(self.q_next.estimator,
                  feed_dict={self.q_next.observations: observations})

class Network():
    def __init__(self, n_features=2, n_actions=4, lr=0.0005, trainable = False):
        #self.default_nl=tf.nn.relu
        self.lr = lr
        self.next_layer_id = 0
        self.n_features = n_features
        self.n_actions = n_actions
        self.theta = {}
        self.estimator = self.vanilla_network()
        self.q_target = tf.placeholder(tf.float32, [None, n_actions])
        if trainable:
            print(self.q_target,'--------aaaaaaaaaaaa--------', self.estimator)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.estimator))
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.sess = None

    def get_layer_id(self):
        this_layer_id = self.next_layer_id
        self.next_layer_id +=1
        return this_layer_id

    def vanilla_network(self, layer_size = [2, 20,20,20, 20,20,20, 4]):
        next_l = self.input_layer() #todo currently the  number of features in the input layer is defined elsewhere
        self.observations = next_l
        for ll, ll_size  in enumerate(layer_size[1:-1]):
            next_l = self.dense_ff_layer(next_l, ll_size)
        ll_size=layer_size[-1]
        next_l = self.dense_ff_layer(next_l, ll_size, nl= lambda x: x)
        return next_l


    def dense_ff_layer(self, previous_layer, output_size, nl=tf.nn.tanh, theta = None):
        if theta is None:
            this_theta = {}
            # print(np.float(np.shape(previous_layer)[-1])**0.5)
            this_theta['w'] = tf.Variable(
                tf.random_normal(shape=[np.shape(previous_layer)[-1].value, output_size],
                                 mean=0.0,
                                 stddev=3.0 / np.sqrt(np.shape(previous_layer)[-1].value)))
            this_theta['b'] = tf.Variable(
                tf.random_normal(shape=[1, output_size],
                                 mean=0.0,
                                 stddev=0.01))
        else:
            error('explicit theta is still unsupported')
        self.theta[self.get_layer_id()] = this_theta
        #print(self.get_layer_id())
        ff_layer = nl(tf.matmul(previous_layer, this_theta['w']) + this_theta['b'])
        return ff_layer

    def input_layer(self):
        return tf.placeholder(tf.float32, [None, self.n_features])

    def train_step_op(self):
        return tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def assign_param_prep(self,source_nwk): #todo support more elaborated structures than double dictionary
        self.assign_param_op = []
        for ll in source_nwk.theta.keys():
            for this_param in source_nwk.theta[ll]:
                self.assign_param_op.append(tf.assign(self.theta[ll][this_param],
                                                 source_nwk.theta[ll][this_param]))
    def update(self, sess):
        sess.run(self.assign_param_op)

