
import numpy as np
import time
import sys
from misc import HP

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

class Scene():
    def __init__(self,image_matrix = None):
        self.image = image_matrix
        self.maxy, self.maxx = np.shape(image_matrix)
        self.hp = HP()

    def edge_image_x(self,edge_location,contrast=1.0):
        self.image = np.zeros(self.image.shape)
        self.image[:,int(edge_location)] = contrast

class Sensor():
    def __init__(self):
        self.hp = HP
        self.hp.winx = 10
        self.hp.winy = 10
        self.frame_size = self.hp.winx * self.hp.winy
        self.frame_view = np.zeros([self.hp.winy,self.hp.winx])
        self.dvs_view =self.dvs_fun(self.frame_view, self.frame_view)

    def update(self,scene,agent):
        current_view = self.get_view(scene, agent)
        self.dvs_view =self.dvs_fun(current_view, self.frame_view)
        self.frame_view = current_view

    def dvs_fun(self,current_frame, previous_frame):
        return current_frame - previous_frame

    def get_view(self,scene,agent):
        return scene.image[scene.maxy - agent.q[1] - self.hp.winy: scene.maxy - agent.q[1],
               agent.q[0]: agent.q[0]+self.hp.winx]

class Agent():
    def __init__(self,max_q = None):
        self.hp = HP()
        #self.hp.action_space = [-2,-1,0,1,2]
        self.hp.action_space = ['v_right','v_left','null'] #'
        self.hp.returning_force = 0.001 #0.00001
        self.max_q = max_q
        self.q_centre = np.array(self.max_q, dtype='f') / 2
        self.q_ana = np.array([np.random.randint(self.max_q[0]), np.random.randint(self.max_q[1])], dtype='f')
        self.qdot = np.array([0.0,0.0])
        self.qdotdot = np.array([0.0,0.0])
        self.q = np.int32(np.floor(self.q_ana))


    def act(self,a):
        if a is None:
            action = 'null'
        else:
            action = self.hp.action_space[a]
        #delta_a = 0.001
        if type(action)==int:
            self.qdot[0] = action
            self.qdotdot = np.array([0., 0.])
        elif action == 'v_up':   # up
            self.qdot[1] = self.qdot[1] + 1 if self.q[1] < self.max_q[1]-1 else -0
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

        #print('debug', self.max_q, self.q_centre)
        self.qdot += self.qdotdot
        #self.qdot -= self.hp.returning_force*(self.q_ana-self.q_centre)
        self.q_ana +=self.qdot
        self.q_ana = np.minimum(self.q_ana,self.max_q)
        self.q_ana = np.maximum(self.q_ana,[0.0,0.0])
        self.q = np.int32(np.floor(self.q_ana))


class Rewards():
    def __init__(self,reward_types=['binary_intensity', 'speed'],relative_weights=[1.0,-0.01]):
        self.reward_obj_list = []
        self.hp=HP()
        self.hp.reward_types = reward_types
        self.hp.relative_weights = relative_weights
        self.reward = 0
        self.rewards = np.zeros([len(reward_types)])
        self.hp.reward_hp = {}
        for reward_type in self.hp.reward_types:
            if reward_type == 'reconstruct':
                this_reward = self.Reconstruction_reward()
            elif reward_type == 'rms_intensity':
                this_reward = self.RMS_intensity_reward()
            elif reward_type == 'binary_intensity':
                this_reward = self.Binary_intensity_reward()
            elif reward_type == 'speed':
                this_reward = self.Speed_reward()
            else:
                error('unknown reward')
            self.reward_obj_list.append(this_reward)
            self.hp.reward_hp[reward_type] = this_reward.hp

    def update_rewards(self, sensor = None, agent = None):
        for ii,this_reward in enumerate(self.reward_obj_list):
            this_reward.update(sensor = sensor,agent = agent)
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

        def update(self, sensor = None, agent = None):
            self.reward = self.hp.alpha_decay*np.sqrt(np.mean(sensor.dvs_view ** 2)) + (1-self.hp.alpha_decay)*self.reward

    class Binary_intensity_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0
            self.th = 1e-3

        def update(self, sensor = None, agent = None):
            self.reward = self.hp.alpha_decay*(np.sqrt(np.mean(sensor.dvs_view ** 2))>self.th) + (1-self.hp.alpha_decay)*self.reward

    class Speed_reward():
        def __init__(self):
            self.hp = HP()
            self.hp.alpha_decay = 1.0
            self.reward = 0

        def update(self, sensor = None, agent = None):
            self.reward = self.hp.alpha_decay*np.sqrt(np.mean(agent.qdot**2)) + (1-self.hp.alpha_decay)*self.reward