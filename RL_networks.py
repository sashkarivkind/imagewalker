import numpy as np
import tensorflow as tf
import pickle


class HP():
    pass

class DQN_net():
    def __init__(self,n_features, n_actions, learning_rate, arch='mlp'):
        self.q_eval = Network(n_features, n_actions, arch=arch, trainable=True, lr=learning_rate)
        self.q_next = Network(n_features, n_actions, arch=arch)
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

    def eval_eval(self,observations):   #this one replaces previous eval method, renamed to avoid reserved words.
        return self.sess.run(self.q_eval.estimator,
                  feed_dict={self.q_eval.observations: observations})

    def eval_incl_layers(self,observations):
        return self.sess.run([self.q_eval.estimator]+[self.q_eval.layers[uu] for uu in sorted(self.q_eval.layers.keys())],
                  feed_dict={self.q_eval.observations: observations})

    def eval_next(self,observations):
        return  self.sess.run(self.q_next.estimator,
                  feed_dict={self.q_next.observations: observations})

    def save_nwk_param(self, filename):
        with self.sess.as_default():
            with open(filename, 'wb') as f:
                pickle.dump([self.q_eval.theta_values(), self.q_next.theta_values()], f)

    def load_nwk_param(self, filename):
        with self.sess.as_default():
            with open(filename, 'rb') as f:
                theta_list = pickle.load(f)
                self.q_eval.theta_update(theta_list[0])
                self.q_next.theta_update(theta_list[1])

class Network():
    def __init__(self, n_features, n_actions, lr=None, trainable = False, arch='mlp', layer_size=None):
        print('debug n_features:',n_features)
        self.hp = HP()
        #self.default_nl=tf.nn.relu
        self.hp.lr = lr
        self.next_layer_id = 0
        self.layers = {}
        self.n_features = n_features
        self.n_actions = n_actions
        self.theta = {}
        self.hp.arch = arch
        default_layer_size={'mlp':  [None]+[400]+[200]*3+[ None],
                            'conv': [None]+[[3,3,32]]+[[2,2,16]]+[200]+[ None]}

        layer_size = default_layer_size[arch] if layer_size is None else layer_size

        if arch == 'mlp':
            self.estimator = self.vanilla_network(layer_size=layer_size)
        elif arch == 'conv':
            self.estimator = self.conv_network(layer_size=layer_size)
        else:
            error
        self.hp.layer_size =layer_size

        self.q_target = tf.placeholder(tf.float32, [None, n_actions])
        if trainable:
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.estimator))
            # self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.q_target, self.estimator))
            # tf.losses.absolute_difference()
            # self.train_op = tf.train.GradientDescentOptimizer(self.hp.lr).minimize(self.loss)
            # self.train_op = tf.train.RMSPropOptimizer(self.hp.lr).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer(self.hp.lr).minimize(self.loss)
        self.sess = None

    def get_layer_id(self):
        this_layer_id = self.next_layer_id
        self.next_layer_id +=1
        return this_layer_id

    def vanilla_network(self, layer_size = [None]+[400]+[200]*3+[ None]):
        layer_size[0] = self.n_features
        layer_size[-1] = self.n_actions
        next_l = self.input_layer() #todo currently the  number of features in the input layer is defined elsewhere
        self.observations = next_l
        for ll, ll_size  in enumerate(layer_size[1:-1]):
            next_l = self.dense_ff_layer(next_l, ll_size)
            # next_l = tf.nn.dropout(next_l, 0.95)
        ll_size=layer_size[-1]
        next_l = self.dense_ff_layer(next_l, ll_size, nl= lambda x: x, g=1e-10)
        return next_l

    def conv_network(self, layer_size = [None]+[[3,3,32]]+[[2,2,16]]+[200]+[ None]):
        layer_size[0] = self.n_features
        layer_size[-1] = self.n_actions
        next_l = self.input_layer2d() #todo currently the  number of features in the input layer is defined elsewhere
        self.observations = next_l
        for ll, ll_size  in enumerate(layer_size[1:-1]):
            if type(ll_size)==list:
                next_l = self.conv2d_layer(next_l, ll_size)
                next_l = tf.nn.max_pool(next_l,[1,3,3,1],[1,1,1,1],'SAME')
            else:
                next_l = self.dense_ff_layer(tf.contrib.layers.flatten(next_l), ll_size)
        ll_size=layer_size[-1]
        next_l = self.dense_ff_layer(next_l, ll_size, nl= lambda x: x, g=1e-10)
        return next_l

    def dense_ff_layer(self, previous_layer, output_size, nl=tf.nn.tanh, theta = None,g=1.0):
        if theta is None:
            this_theta = {}
            # print('debug:',np.float(np.shape(previous_layer)[-1])**0.5)
            print('debug:',np.shape(previous_layer))
            this_theta['w'] = tf.Variable(
                tf.random_normal(shape=[np.shape(previous_layer)[-1].value, output_size],
                                 mean=0.0,
                                 stddev=g*2.0 / np.sqrt(np.shape(previous_layer)[-1].value)))
            this_theta['b'] = tf.Variable(
                tf.random_normal(shape=[1, output_size],
                                 mean=0.0,
                                 stddev=0.01))
        else:
            error('explicit theta is still unsupported')
        layer_id=self.get_layer_id()
        self.theta[layer_id] = this_theta
        self.layers[layer_id] = nl(tf.matmul(previous_layer, this_theta['w']) + this_theta['b'])
        return self.layers[layer_id]

    def conv2d_layer(self, previous_layer, filters_hwc, nl=tf.nn.relu, theta=None, g=1.0):
        if theta is None:
            this_theta = {}
            print('debu:', [np.shape(previous_layer)[1].value,np.shape(previous_layer)[2].value,filters_hwc[2]])
            this_theta['w'] = tf.Variable(
                tf.random_normal(shape=filters_hwc[:2]+[np.shape(previous_layer)[-1].value,filters_hwc[2]],
                                 mean=0.0,
                                 stddev=g / np.sqrt(filters_hwc[-1]*filters_hwc[-2])))
        else:
            error('explicit theta is still unsupported')
        layer_id=self.get_layer_id()
        self.theta[layer_id] = this_theta
        self.layers[layer_id] =  tf.nn.conv2d(
                previous_layer,
                this_theta['w'],
                [1,2,2,1], #todo: generalize
                "SAME"
        )

        bias_shape=[uu.value for uu in np.shape(self.layers[layer_id])]
        bias_shape[0]=1

        this_theta['b'] = tf.Variable(
            tf.random_normal(
                shape=bias_shape,
                mean=0.0,
                stddev=0.01))

        self.layers[layer_id] +=this_theta['b']
        self.layers[layer_id] += nl(self.layers[layer_id])
        return self.layers[layer_id]

    def input_layer(self):
        return tf.placeholder(tf.float32, [None, self.n_features])

    def input_layer2d(self):
        return tf.placeholder(tf.float32, [None] + self.n_features)

    # def train_step_op(self):
    #     return tf.train.RMSPropOptimizer(self.hp.lr).minimize(self.loss)

    def assign_param_prep(self,source_nwk): #todo support more elaborated structures than double dictionary
        self.assign_param_op = []
        for ll in source_nwk.theta.keys():
            for this_param in source_nwk.theta[ll]:
                self.assign_param_op.append(tf.assign(self.theta[ll][this_param],
                                                       source_nwk.theta[ll][this_param]))

    def theta_values(self): #todo support more elaborated structures than double dictionary
        t = {}
        for ll in self.theta.keys():
            t[ll] = {}
            for this_param in self.theta[ll]:
                t[ll][this_param] = self.theta[ll][this_param].eval(self.sess)
        return t

    def theta_update(self,t): #todo support more elaborated structures than double dictionary
        for ll in t.keys():
            for this_param in t[ll]:
                self.theta[ll][this_param].assign(t[ll][this_param]).op.run(session=self.sess)

    def update(self, sess):
        sess.run(self.assign_param_op)
