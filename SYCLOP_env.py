
import numpy as np
import time
import sys
from misc import HP

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

class Scene():
    def __init__(self,image_matrix = None, frame_list = None ):
        if image_matrix is not None:
            self.image = image_matrix
        if frame_list is not None:
            self.movie_mode = True
            self.current_frame = 0
            self.frame_list = frame_list
            self.total_frames = len(self.frame_list)
            self.image = self.frame_list[self.current_frame]
        self.maxy, self.maxx = np.shape(self.image)
        self.hp = HP()

    def edge_image_x(self,edge_location,contrast=1.0):
        self.image = np.zeros(self.image.shape)
        self.image[:,int(edge_location)] = contrast

    def update(self):
        if self.movie_mode:
            self.current_frame = (self.current_frame+1)%self.total_frames
            self.image = self.frame_list[self.current_frame]
        else:
            error #not implemented yet

class Sensor():
    def __init__(self, log_mode=False, log_floor = 1e-3, fading_mem=0.0):
        self.hp = HP
        self.hp.winx = 16*4
        self.hp.winy = 16*4
        self.hp.centralwinx = 4*4
        self.hp.centralwiny = 4*4
        self.log_mode = log_mode
        self.log_floor = log_floor
        self.hp.fading_mem = fading_mem


        self.cwx1 = (self.hp.winx-self.hp.centralwinx)//2
        self.cwy1 = (self.hp.winy-self.hp.centralwiny)//2
        self.cwx2 = self.cwx1 + self.hp.centralwinx
        self.cwy2 = self.cwy1 + self.hp.centralwiny
        self.frame_size = self.hp.winx * self.hp.winy
        self.reset()

    def reset(self):
        self.frame_view = np.zeros([self.hp.winy,self.hp.winx])
        self.central_frame_view = self.frame_view[self.cwy1:self.cwy2,self.cwx1:self.cwx2]
        self.dvs_view =self.dvs_fun(self.frame_view, self.frame_view)
        self.central_dvs_view =self.dvs_view[self.cwy1:self.cwy2,self.cwx1:self.cwx2]


    def update(self,scene,agent):
        current_view = self.get_view(scene, agent)
        self.dvs_view = self.dvs_view*self.hp.fading_mem+self.dvs_fun(current_view, self.frame_view)
        self.central_dvs_view =self.dvs_view[self.cwy1:self.cwy2,self.cwx1:self.cwx2]
        self.frame_view = current_view
        self.central_frame_view = self.frame_view[self.cwy1:self.cwy2,self.cwx1:self.cwx2]

    def dvs_fun(self,current_frame, previous_frame):
        delta = current_frame - previous_frame
        if self.log_mode:
            return (np.log10(np.abs(delta)+self.log_floor)-np.log(self.log_floor))*np.sign(delta)
        else:
            return current_frame - previous_frame

    def get_view(self,scene,agent):
        return scene.image[scene.maxy - agent.q[1] - self.hp.winy: scene.maxy - agent.q[1],
               agent.q[0]: agent.q[0]+self.hp.winx]

class Agent():
    def __init__(self,max_q = None):
        self.hp = HP()
        # self.hp.action_space =[-1,1]# [-3,-2,-1,0,1,2,3]
        # self.hp.action_space = ['v_right','v_left','v_up','v_down','null'] #'            #,'R','L','U','D'] +
        # self.hp.action_space = ['v_right','v_left','v_up','v_down','null','R','L','U','D'] + \
        #                       [['v_right','v_up'],['v_right','v_down'],['v_left','v_up'],['v_left','v_down']]#'
        self.hp.action_space = ['v_right','v_left','v_up','v_down','null'] + \
                              [['v_right','v_up'],['v_right','v_down'],['v_left','v_up'],['v_left','v_down']]#'
        self.hp.big_move = 25
        self.max_q = max_q
        self.q_centre = np.array(self.max_q, dtype='f') / 2
        self.saccade_flag = False
        self.reset()

    def reset(self):
        self.q_ana = np.array([np.random.randint(self.max_q[0]), np.random.randint(self.max_q[1])], dtype='f')
        self.qdot = np.array([0.0,0.0])
        self.qdotdot = np.array([0.0,0.0])
        self.q = np.int32(np.floor(self.q_ana))


    def act(self,a):
        if a is None:
            action = 'null'
        else:
            action = self.hp.action_space[a]

        self.saccade_flag = False

        #delta_a = 0.001
        if type(action) == list:
            for subaction in action:
                self.parse_action(subaction)
        else:
            self.parse_action(action)

        #print('debug', self.max_q, self.q_centre)
        self.qdot += self.qdotdot
        #self.qdot -= self.hp.returning_force*(self.q_ana-self.q_centre)
        self.q_ana +=self.qdot
        self.enforce_boundaries()

    def enforce_boundaries(self):
        self.q_ana = np.minimum(self.q_ana,self.max_q)
        self.q_ana = np.maximum(self.q_ana,[0.0, 0.0])
        self.q = np.int32(np.floor(self.q_ana))

    def parse_action(self,action):
        if type(action)==int: #todo  - int actions denote velocity shift of velocity in x direction. this needs to be generalized
            self.qdot[0] = action
            self.qdotdot = np.array([0., 0.])
        elif action == 'reset':
            self.reset()
            self.saccade_flag = True
        elif action == 'R':
            self.q_ana[0] += self.hp.big_move
            self.qdot = np.array([0.0, 0.0])
            self.qdotdot = np.array([0.0, 0.0])
            self.saccade_flag = True
            self.enforce_boundaries()

        elif action == 'L':
            self.q_ana[0] -= self.hp.big_move
            self.qdot = np.array([0.0, 0.0])
            self.qdotdot = np.array([0.0, 0.0])
            self.saccade_flag = True
            self.enforce_boundaries()

        elif action == 'U':
            self.q_ana[1] += self.hp.big_move
            self.qdot = np.array([0.0, 0.0])
            self.qdotdot = np.array([0.0, 0.0])
            self.saccade_flag = True
            self.enforce_boundaries()

        elif action == 'D':
            self.q_ana[1] -= self.hp.big_move
            self.qdot = np.array([0.0, 0.0])
            self.qdotdot = np.array([0.0, 0.0])
            self.saccade_flag = True
            self.enforce_boundaries()

        elif action == 'v_up':  # up
            self.qdot[1] = self.qdot[1] + 1 if self.q[1] < self.max_q[1] - 1 else -0
            self.qdotdot = np.array([0., 0.])
        elif action == 'v_down':   # down
            self.qdot[1] = self.qdot[1] - 1 if self.q[1] > 0 else -0
            self.qdotdot = np.array([0., 0.])
        elif action == 'v_left':   # left
            self.qdot[0] = self.qdot[0]-1 if self.q[0] > 0 else -0
            self.qdotdot = np.array([0., 0.])
        elif action == 'v_right':   # right
            self.qdot[0] = self.qdot[0]+1 if self.q[0] < self.max_q[0]-1 else -0
            self.qdotdot = np.array([0.,0.])
        elif action == 'a_up':   # up
            self.qdotdot[1] = self.qdotdot[1] + delta_a if self.q[1] < self.max_q[1]-1 else -0
        elif action == 'a_down':   # down
            self.qdotdot[1] = self.qdotdot[1] - delta_a if self.q[1] > 0 else -0
        elif action == 'a_left':   # left
            self.qdotdot[0] = self.qdotdot[0] - delta_a if self.q[0] > 0 else -0
        elif action == 'a_right':   # right
            self.qdotdot[0] = self.qdotdot[0] + delta_a if self.q[0] < self.max_q[0]-1 else -0
        elif action == 'null':   # null
            pass
        else:
            error('unknown action')

class Rewards():
    def __init__(self,reward_types=['central_rms_intensity'],relative_weights=[1.0]):
        self.reward_obj_list = []
        self.hp=HP()
        self.hp.reward_types = reward_types
        self.hp.relative_weights = relative_weights
        self.reset()

    def reset(self):
        self.reward = 0
        self.rewards = np.zeros([len(self.hp.reward_types)])
        self.hp.reward_hp = {}
        for reward_type in self.hp.reward_types:
            if reward_type == 'reconstruct':
                this_reward = self.Reconstruction_reward()
            elif reward_type == 'rms_intensity':
                this_reward = self.RMS_intensity_reward()
            elif reward_type == 'central_homeostatic_intensity':
                this_reward = self.Central_homeostatic_intensity_reward()
            elif reward_type == 'central_rms_intensity':
                this_reward = self.Central_RMS_intensity_reward()
            elif reward_type == 'binary_intensity':
                this_reward = self.Binary_intensity_reward()
            elif reward_type == 'central_binary_intensity':
                this_reward = self.Central_binary_intensity_reward()
            elif reward_type == 'debug_central_binary_intensity':
                this_reward = self.Debug_binary_intensity_reward()
            elif reward_type == 'speed':
                this_reward = self.Speed_reward()
            elif reward_type == 'boundaries':
                this_reward = self.Boundary_reward()
            elif reward_type == 'saccade':
                this_reward = self.Saccade_reward()
            elif reward_type == 'network':
                this_reward = self.Network_intensity_reward()
            else:
                error('unknown reward')
            self.reward_obj_list.append(this_reward)
            self.hp.reward_hp[reward_type] = this_reward.hp

    def update_rewards(self, sensor=None, agent=None, network=None):
        for ii,this_reward in enumerate(self.reward_obj_list):
            this_reward.update(sensor = sensor,agent = agent, network = network)
            self.rewards[ii] = this_reward.reward
        self.reward = np.sum(self.hp.relative_weights*self.rewards)

    class Reconstruction_reward():
        def __init__(self):
            error('under rebuilding, don\'t use')
            self.reconstruction = np.zeros([IMAGE_Y, IMAGE_X])

        def update(self,sensor = None, agent = None):
            self.copy_current_region()
            self.reward = -np.sqrt(np.mean((self.reconstruction - self.orig) ** 2))
            self.reconstruction *= RECONSTRUCT_DECAY
        def copy_current_region(self):
            max_x = np.min([self.pos['x']+WIN_X_HLF,IMAGE_X])
            max_y = np.min([self.pos['y']+WIN_Y_HLF,IMAGE_Y])
            min_x = np.max([self.pos['x']-WIN_X_HLF,0])
            min_y = np.max([self.pos['y']-WIN_Y_HLF,0])
            #approaching matrix coordinates with y first and y counted from above requires some care
            self.reconstruction[IMAGE_Y-max_y:IMAGE_Y-min_y,min_x:max_x] = self.orig[IMAGE_Y-max_y:IMAGE_Y-min_y,min_x:max_x]

    class RMS_intensity_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0

        def update(self, sensor = None, agent = None, network=None):
            self.reward = self.hp.alpha_decay*np.sqrt(np.mean(sensor.dvs_view ** 2)) + (1-self.hp.alpha_decay)*self.reward

    class Central_RMS_intensity_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0
            self.th = 1e-3

        def update(self, sensor = None, agent = None, network=None):
            self.reward = self.hp.alpha_decay*np.sqrt(np.mean((1.0*sensor.central_dvs_view) ** 2))  + (1-self.hp.alpha_decay)*self.reward

    class Central_L1_intensity_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0
            self.th = 1e-3

        def update(self, sensor=None, agent=None, network=None):
            self.reward = self.hp.alpha_decay * np.mean((1.0 * np.abs(sensor.central_dvs_view) ** 2)) + (1 - self.hp.alpha_decay) * self.reward

    class Central_homeostatic_intensity_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.hp.target_intensity = 0.5
            self.reward = 0
            self.th = 1e-3

        def update(self, sensor = None, agent = None, network=None):
            self.reward = self.hp.alpha_decay*np.sqrt(np.mean(sensor.central_dvs_view ** 2)) + (1-self.hp.alpha_decay)*self.reward
            self.reward = -np.abs(self.reward-self.hp.target_intensity)

    class Binary_intensity_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0
            self.th = 1e-3

        def update(self, sensor = None, agent = None, network=None):
            self.reward = self.hp.alpha_decay*(np.sqrt(np.mean(sensor.dvs_view ** 2))>self.th) + (1-self.hp.alpha_decay)*self.reward

    class Speed_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0

        def update(self, sensor = None, agent = None, network=None):
            self.reward = self.hp.alpha_decay*np.sqrt(np.mean(agent.qdot**2)) + (1-self.hp.alpha_decay)*self.reward

    class Saccade_reward():
        def __init__(self):
            self.hp = HP()

        def update(self, sensor = None, agent = None, network=None):
            self.reward = 1.0*agent.saccade_flag

    class Boundary_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0

        def update(self, sensor = None, agent = None, network=None):
            self.reward = self.hp.alpha_decay*( (agent.q_ana[0] < 1) or (agent.q_ana[0]>agent.max_q[0]-1) or (agent.q_ana[1] < 1) or (agent.q_ana[1]>agent.max_q[1]-1))\
                          + (1-self.hp.alpha_decay)*self.reward

    class Central_binary_intensity_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0
            self.th = 1e-3

        def update(self, sensor = None, agent = None, network=None):
            self.reward = self.hp.alpha_decay*(np.sqrt(np.mean(sensor.central_dvs_view ** 2))>self.th) + (1-self.hp.alpha_decay)*self.reward

    class Debug_binary_intensity_reward(): #just
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0
            self.th = 1e-3

        def update(self, sensor = None, agent = None, network=None):
            self.reward = self.hp.alpha_decay*(np.sqrt(np.var(sensor.central_frame_view))>self.th) + (1-self.hp.alpha_decay)*self.reward

    class Network_intensity_reward(): #just
        def __init__(self):
            self.hp = HP()

        def update(self, sensor = None, agent = None, network=None):
            self.reward = np.sqrt(np.mean(network**2))