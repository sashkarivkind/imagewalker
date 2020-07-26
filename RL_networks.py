import numpy as np
import tensorflow as tf
import pickle


class HP():
    pass

class DQN_net():
    def __init__(self,n_features, n_actions, learning_rate, arch='mlp',**kwargs):
        #todo: NB!!! argument n_features used here is called n_features_shaped in the upper hierarchy
        #in the upper hierarchy n_feature is a scalar, representing the total number of features
        self.q_eval = Stand_alone_net(n_features, n_actions, lr=learning_rate, arch=arch, trainable=True,**kwargs)
        self.q_next = Stand_alone_net(n_features, n_actions, arch=arch, trainable=False,**kwargs)
        self.sess = None


    def reset(self):
        self.q_next.net.assign_param_prep(self.q_eval.net) #todo move to parent object
        self.update_q_next()

    def assign_session_to_nwk(self, sess):
        self.sess = sess
        self.q_next.assign_session_to_nwk(sess)
        self.q_eval.assign_session_to_nwk(sess)

    def update_q_next(self):
        self.q_next.net.update(self.sess)  #todo move to parent object

    def training_step(self, observations, q_target):
        return self.q_eval.training_step(observations, q_target)

    def eval_eval(self,observations):   #this one replaces previous eval method, renamed to avoid reserved words.
        return self.q_eval.eval_eval(observations)

    def eval_incl_layers(self,observations):
        return self.q_eval.eval_incl_layers(observations)


    def eval_next(self,observations):
        return self.q_next.eval_eval(observations)


    def save_nwk_param(self, filename):
        with self.sess.as_default():
            with open(filename, 'wb') as f:
                pickle.dump([self.q_eval.net.theta_values(), self.q_next.net.theta_values()], f) #todo move to parent object

    def load_nwk_param(self, filename):
        with self.sess.as_default():
            with open(filename, 'rb') as f:
                theta_list = pickle.load(f)
                self.q_eval.net.theta_update(theta_list[0]) #todo move to parent object
                self.q_next.net.theta_update(theta_list[1])




class Stand_alone_net(): #wrapper for network
    def __init__(self,n_features, n_outputs, **kwargs):
        self.net = Network(n_features, n_outputs, **kwargs)
        self.sess = None

    def observation_receiver(self, observations):
        #this method is designed to organize inputs from higher levels. This is in addition to a higher level shaping made by "shape_fun" method
        if type(observations)==list:
            return {self.net.inputs[ii]: observation for ii,observation in enumerate(observations)}
        else:
            return {self.net.inputs[0]:observations}

    def assign_session_to_nwk(self, sess):
        print('debug SESSSS!!!!')
        self.sess = sess
        self.net.sess = sess

    def training_step(self, observations, q_target):
        feed_dict = self.observation_receiver(observations)
        feed_dict.update({self.net.q_target: q_target})
        return self.net.sess.run([self.net.train_op,self.net.loss],
                  feed_dict=feed_dict)

    def eval_eval(self,observations):   #this one replaces previous eval method, renamed to avoid reserved words.
        return self.net.sess.run(self.net.estimator,
                  feed_dict=self.observation_receiver(observations))

    def eval_incl_layers(self,observations):
        return self.net.sess.run([self.net.estimator]+[self.net.layers[uu] for uu in sorted(self.net.layers.keys())]+[self.net.reg_term],
                  feed_dict=self.observation_receiver(observations))

    def save_nwk_param(self, filename):
        with self.net.sess.as_default():
            with open(filename, 'wb') as f:
                pickle.dump(self.net.theta_values(), f)

    def load_nwk_param(self, filename):
        with self.net.sess.as_default():
            with open(filename, 'rb') as f:
                theta = pickle.load(f)
                self.net.theta_update(theta)


class Network():
    def __init__(self, n_features, n_actions, **kwargs):

        default_kwargs = {'n_ctrls': 2,
                            'lr': None,
                            'trainable' : False,
                            'arch': 'mlp',
                            'layer_size' : None,
                            'optimizer' : 'Adam',
                            'loss_type' : 'mean_squared',
                            'lambda_reg' : 0.0,
                            'scale_layer_en' : True,
                            'train_starting_from_layer' : None
                          }

        default_layer_size={'mlp':  [None]+[400]*1+[200]*3+[10,10]*0+[ None],
                            'conv': [None]+[[3,3,32]]+[[2,2,16]]+[200]+[ None],
                            'conv_ctrl': [None] + [[3, 3, 32]] + [[2, 2, 16]] + [200] + [None],
                            'conv_ctrl_fade_v1': [None] + [[3, 3, 32]] + [[2, 2, 16]] + [200] + [None]}

        print('debug n_features:',n_features)
        self.hp = HP()
        self.hp.__dict__ = {kk:(kwargs[kk] if kk in kwargs.keys() else default_kwargs[kk]) for kk in default_kwargs.keys()}
        self.hp.layer_size = default_layer_size[self.hp.arch] if self.hp.layer_size is None else self.hp.layer_size

        self.next_layer_id = 0
        self.layers = {}
        self.n_features = n_features
        self.n_actions = n_actions
        self.theta = {}
        self.reg_counter_l1=0
        self.inputs=[]

        if self.hp.arch == 'mlp':
            self.estimator = self.vanilla_network(layer_size=self.hp.layer_size)
        elif self.hp.arch == 'conv':
            self.estimator = self.conv_network(layer_size=self.hp.layer_size)
        elif self.hp.arch == 'conv_ctrl':
            self.estimator = self.conv_network_with_ctrls(layer_size=self.hp.layer_size)
        elif self.hp.arch == 'conv_ctrl_fade_v1':
            self.estimator = self.conv_network_with_ctrls_and_fading_v1(layer_size=self.hp.layer_size)

        else:
            error

        if self.hp.optimizer == 'GradientDescent':
            optimizer_fun =  tf.train.GradientDescentOptimizer
        elif self.hp.optimizer == 'RMSprom':
            optimizer_fun = tf.train.RMSPropOptimizer
        elif self.hp.optimizer == 'Adam':
            optimizer_fun = tf.train.AdamOptimizer
        else:
            error

        #todo organize regularization term assingment
        self.reg_term=self.reg_counter_l1

        self.q_target = tf.placeholder(tf.float32, [None, n_actions])
        if self.hp.trainable:
            if self.hp.loss_type=='mean_squared':
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.estimator))
            elif self.hp.loss_type=='softmax_cross_entropy':
                self.loss = tf.losses.softmax_cross_entropy(self.q_target, self.estimator)

            if self.hp.train_starting_from_layer is None:
                self.train_op = optimizer_fun(self.hp.lr).minimize(self.loss + self.hp.lambda_reg * self.reg_term)
            else: #original restricted training took:  [theta[2]['w'], theta[2]['b'], theta[3]['w'], theta[3]['b']]
                theta = self.theta #big_todo!!!!!!
                to_train_list = []
                for ii in range(self.hp.train_starting_from_layer, len(theta)):
                    if type(theta[ii]) is dict:
                        for kk in theta[ii].keys():
                            to_train_list.append(theta[ii][kk])
                    else:
                        to_train_list.append(theta[ii])
                self.train_op = optimizer_fun(self.hp.lr).minimize(self.loss+self.hp.lambda_reg*self.reg_term,var_list=to_train_list)

        self.sess = None

    def get_layer_id(self):
        this_layer_id = self.next_layer_id
        self.next_layer_id +=1
        return this_layer_id

    def vanilla_network(self, layer_size = [None]+[400]+[200]*3+[ None]):
        layer_size[0] = self.n_features
        layer_size[-1] = self.n_actions
        next_l = self.input_layer() #todo currently the  number of features in the input layer is defined elsewhere
        self.inputs.append(next_l)
        for ll, ll_size  in enumerate(layer_size[1:-1]):
            next_l = self.dense_ff_layer(next_l, ll_size)
            # next_l = tf.nn.dropout(next_l, 0.95)
        ll_size=layer_size[-1]
        next_l = self.dense_ff_layer(next_l, ll_size, nl= lambda x: x, g=1e-10)
        if self.hp.scale_layer_en:
            next_l = self.scaling_layer(next_l)
        return next_l

    def conv_network(self, layer_size = [None]+[[3,3,32]]+[[2,2,16]]+[200]+[ None]):
        layer_size[0] = self.n_features
        layer_size[-1] = self.n_actions
        next_l = self.input_layer2d() #todo currently the  number of features in the input layer is defined elsewhere
        self.inputs.append(next_l)
        for ll, ll_size  in enumerate(layer_size[1:-1]):
            if type(ll_size)==list:
                next_l = self.conv2d_layer(next_l, ll_size)
                self.reg_counter_l1+=tf.reduce_mean(tf.abs(next_l))
                next_l = tf.nn.max_pool(next_l,[1,3,3,1],[1,1,1,1],'SAME')
            else:
                next_l = self.dense_ff_layer(tf.contrib.layers.flatten(next_l), ll_size)
        ll_size=layer_size[-1]
        next_l = self.dense_ff_layer(next_l, ll_size, nl= lambda x: x, g=1e-10)
        return next_l

    def conv_network_with_ctrls(self, layer_size = [None]+[[3,3,32]]+[[2,2,16]]+[200]+[ None]):
        #an architechture of cnn with control signals merged at the first dense layer
        layer_size[0] = self.n_features
        layer_size[-1] = self.n_actions
        next_l = self.input_layer2d() #todo currently the  number of features in the input layer is defined elsewhere
        self.inputs.append( next_l )#visual input
        self.inputs.append(self.input_layer(size=self.hp.n_ctrls) ) #controls, i.e. speed signal
        merged_ctrls = False
        for ll, ll_size  in enumerate(layer_size[1:-1]):
            if type(ll_size)==list:
                next_l = self.conv2d_layer(next_l, ll_size)
                self.reg_counter_l1+=tf.reduce_mean(tf.abs(next_l))
                next_l = tf.nn.max_pool(next_l,[1,3,3,1],[1,1,1,1],'SAME')
            else: #triggered at the first fully connected layer
                next_l_prime = tf.contrib.layers.flatten(next_l)
                next_l_prime = next_l_prime if merged_ctrls else tf.concat([next_l_prime,self.inputs[1]],1)
                next_l = self.dense_ff_layer(next_l_prime, ll_size)
                merged_ctrls = True
        ll_size=layer_size[-1]
        next_l = self.dense_ff_layer(next_l, ll_size, nl= lambda x: x, g=1e-10)
        return next_l

    def conv_network_with_ctrls_and_fading_v1(self, layer_size = [None]+[[3,3,32]]+[[2,2,16]]+[200]+[ None],layer_with_fading=1):
        #an architechture of cnn with control signals merged at the first dense layer
        #and an additive signal after the last convolutionbal layer, that enables injection of additive signal, used to immitate fading memory
        layer_size[0] = self.n_features
        layer_size[-1] = self.n_actions
        next_l = self.input_layer2d() #todo currently the  number of features in the input layer is defined elsewhere
        self.inputs.append( next_l )#visual input
        self.inputs.append(self.input_layer(size=self.hp.n_ctrls) ) #controls, i.e. speed signal

        merged_ctrls = False
        for ll, ll_size  in enumerate(layer_size[1:-1]):
            if type(ll_size)==list:
                next_l = self.conv2d_layer(next_l, ll_size)
                if ll == layer_with_fading: #todo generalize
                    self.inputs.append(tf.placeholder(tf.float32, next_l.shape))  # externally stored additive term to layer
                    next_l += self.inputs[2]
                self.reg_counter_l1+=tf.reduce_mean(tf.abs(next_l))
                next_l = tf.nn.max_pool(next_l,[1,3,3,1],[1,1,1,1],'SAME')
            else: #triggered at the first fully connected layer
                next_l_prime = tf.contrib.layers.flatten(next_l)
                next_l_prime = next_l_prime if merged_ctrls else tf.concat([next_l_prime,self.inputs[1]],1)
                next_l = self.dense_ff_layer(next_l_prime, ll_size)
                merged_ctrls = True
        ll_size=layer_size[-1]
        next_l = self.dense_ff_layer(next_l, ll_size, nl= lambda x: x, g=1e-10)
        return next_l

    def dense_ff_layer(self, previous_layer, output_size, nl=tf.nn.tanh, theta = None,g=1.0): #todo organize support for multiple nonlinearities
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

    def scaling_layer(self, previous_layer, theta = None):
        if theta is None:
            this_theta = {}
            print('debug:',np.shape(previous_layer))
            this_theta['scale'] = tf.Variable(1.0)
        else:
            error('explicit theta is still unsupported')
        layer_id=self.get_layer_id()
        self.theta[layer_id] = this_theta
        self.layers[layer_id] = previous_layer * this_theta['scale']
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
        # self.layers[layer_id] += nl(self.layers[layer_id]) #NB! todo NOTE: this was a major "issue" in previous version. This is like a leaky relu...
        self.layers[layer_id] = nl(self.layers[layer_id])
        return self.layers[layer_id]

    def input_layer(self, size=None):
        if size is None:
            return tf.placeholder(tf.float32, [None, self.n_features])
        else:
            return tf.placeholder(tf.float32, [None, size])

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


class DQN_net_old():
#newer version is rewritten using Stand_alone_network object
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


