
import numpy as np
import time
import sys
from mnist import MNIST
import pickle

#from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

mnist = MNIST('/home/bnapp/datasets/mnist/')
images,labels = mnist.load_training()


class Image_env1( object):
    def __init__(self,bmp_features = False):

        #self.padded_win = np.zeros(WIN_Y, WIN_X)
        self.pos = {'x':np.random.randint(IMAGE_X) ,'y': np.random.randint(IMAGE_Y)}
        #self.pos = {'x':IMAGE_X // 2,'y':IMAGE_Y // 2}

        self.image_x = IMAGE_X  # image size (set according to MNIST)
        self.image_y = IMAGE_Y
        self.reward_list = []
        self.value = 0
        self.value_alpha = 1.0/10000
        self.value_list = []
        self.q_snapshots = []
        self.reward_plot_handler = None
        self.history_file = 'history/rl_history' + str(time.time())+'.pkl'
        self.bmp_features = bmp_features
        if bmp_features:
            self.n_features = WIN_X*WIN_Y
        else:
            self.n_features = 2
        self.init_scenario()


    def init_scenario(self):
        self.orig=1.0*np.array(images[IMAGE_NUM]).reshape([IMAGE_X,IMAGE_Y])
        self.orig= self.orig/np.max(self.orig)
        self.padded_orig = np.zeros([IMAGE_Y+WIN_Y, IMAGE_X+WIN_X])
        self.padded_orig[WIN_Y_HLF:IMAGE_Y+WIN_Y_HLF, WIN_X_HLF:IMAGE_X+WIN_X_HLF] = self.orig
        self.step_cnt = 0
        self.pos = {'x': np.random.randint(IMAGE_X), 'y': np.random.randint(IMAGE_Y)}

    def reset(self):
        #self.update()
        time.sleep(0.1)
        #self.canvas.delete(self.agentwin)
        self.init_scenario()
        s_ = self.current_features()
        return s_



    



class Scene():
    def __init__(self,image_matrix = None):
        self.image = image_matrix
        self.maxy, self.maxx = np.shape(image_matrix)

class Sensor():
    def __init__(self):
        self.hp = HP
        self.hp.winx = 10
        self.hp.winy = 10
        self.frame_size = self.hp.winx * self.hp.winy

    def update(self,scene,agent):
        current_view = self.get_view(scene, agent)
        self.dvs_view =self.dvs_fun(current_view, self.frame_view)
        self.frame_view = current_view

    def dvs_fun(self,current_frame, previous_frame):
        return current_frame - previous_frame

    def get_view(self,scene,agent):
        return scene.image[scene.maxy - agent.q[1] - self.winy: scene.maxy - agent.q[1],
               agent.q[0]: agent.q[0]+self.winx]

class Agent():
    def __init__(self,max_q = None):
        self.hp = HP()
        self.hp.action_space = ['a_right','a_left']
        self.max_q = max_q
        self.q = np.array([np.random.randint(self.max_q[0]), np.random.randint(self.max_q[1])])
        self.qdot = np.array([0,0])
        self.qdotdot = np.array([0,0])

    def act(self,a):
        action = self.hp.action_space[a]
        if action == 'v_up':   # up
            self.qdot[1] += 1 if self.q[1] < self.max_q[1]-1 else -0
        elif action == 'v_down':   # down
            self.qdot[1] -= 1 if self.q[1] > 0 else -0
        elif action == 'v_left':   # left
            self.qdot[0] -= 1 if self.q[0] > 0 else -0
        elif action == 'v_right':   # right
            self.qdot[0] += 1 if self.q[0] < self.max_q[0]-1 else -0
        elif action == 'a_up':   # up
            self.qdotdot[1] += 1 if self.q[1] < self.max_q[1]-1 else -0
        elif action == 'a_down':   # down
            self.qdotdot[1] -= 1 if self.q[1] > 0 else -0
        elif action == 'a_left':   # left
            self.qdotdot[0] -= 1 if self.q[0] > 0 else -0
        elif action == 'a_right':   # right
            self.qdotdot[0] += 1 if self.q[0] < self.max_q[0]-1 else -0
        else:
            error('unknown action')

        self.qdot += self.qdotdot
        self.q +=self.qdot
        self.q = np.minimum(self.q,self.max_q)
        self.q = np.maximum(self.q,[0,0])


class Rewards():
    def __init__(self,reward_types=['rms_intensity'],relative_weights=1):
        self.reward_obj_list = []
        self.hp=HP()
        self.hp.reward_types = reward_types
        self.hp.relative_weights = relative_weights
        self.total_reward = 0
        self.hp.reward_hp = {}
        for reward_type in reward_types:
            if reward_type == 'reconstruct':
                this_reward = self.Reconstruction_reward
            if reward_type == 'rms_intensity':
                this_reward = self.RMS_intensity_reward
            else:
                error('unknown reward')
            self.reward_obj_list.append(this_reward)
            self.hp.reward_hp[reward_type] = this_reward.hp

    def update_rewards(self,observables):
        for ii,this_reward in enumerate(self.rewards):
            this_reward.update(observables)
            self.rewards[ii] = this_reward.reward
        self.reward = np.matmul(self.hp.relative_weights,self.rewards)

    class Reconstruction_reward():
        def __init__(self):
            error('under rebuilding, don\'t use')
            self.reconstruction = np.zeros([IMAGE_Y, IMAGE_X])

        def update(self,observables):
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
            self.hp.alpha_decay = 0.1
            self.reward = 0

        def update(self, observables):
            self.reward = self.hp.alpha_decay*np.sqrt(np.mean(observables.dvs.framed_view() ** 2)) + (1-self.hp.alpha_decay)*self.reward

