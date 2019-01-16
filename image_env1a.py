"""
Reinforcement learning maze example.
This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

"""
import numpy as np
import time
import sys
import pickle

#from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width
IMAGE_X = 5 #image size (set according to MNIST)
IMAGE_Y = 5
WIN_X = 5 #agents window
WIN_Y = 5
DISP_SCALE = 50
REFLECT_EN = 0
RECONSTRUCT_DECAY=0.9

MAX_STEPS=10000

WIN_X_HLF = (WIN_X-1) // 2
WIN_Y_HLF = (WIN_Y-1) // 2

def magnify_image(img,factor):
    return np.kron(img,np.ones([factor,factor]))

def image_from_np(img, scale=256,size_fac=1):
    return ImageTk.PhotoImage(image=Image.fromarray(scale*magnify_image(img,size_fac)))

class Image_env1( object):
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.graphics_on=False
        # self.title('image reconstructor')
        # self.geometry('{0}x{1}'.format(IMAGE_X * DISP_SCALE, IMAGE_Y * DISP_SCALE))
        self.reconstruction = np.zeros([IMAGE_X, IMAGE_Y])
        self.pos = {'x':np.random.randint(IMAGE_X) ,'y': np.random.randint(IMAGE_Y)}
        #self.pos = {'x':IMAGE_X // 2,'y':IMAGE_Y // 2}
        self.init_scenario()
        self.fig,self.ax = plt.subplots(1) #=plt.figure()
        # self.rect = patches.Rectangle((self.pos['x'], self.pos['y']), 0.1, 0.1, linewidth=1, edgecolor='r',
        #                               facecolor='none')
        # self.ax.add_patch(self.rect)
        self.image_x = IMAGE_X  # image size (set according to MNIST)
        self.image_y = IMAGE_Y
        self.reward_list = []
        self.value = 0
        self.value_alpha = 1.0/1000
        self.q_snapshots = []
        self.reward_plot_handler = None
        self.history_file = 'history/rl_history' + str(time.time())+'.pkl'
        plt.ion()
        plt.pause(0.1)
    def init_scenario(self):
        #self.orig=np.eye(IMAGE_X,IMAGE_Y)
        #self.orig=np.zeros([IMAGE_X,IMAGE_Y])
        #self.orig[1:IMAGE_X-1,1:IMAGE_Y-1]=1
        self.orig=np.ones([IMAGE_X,IMAGE_Y])
        self.orig[1:IMAGE_X-1,1:IMAGE_Y-1]=0
        # self.orig=np.zeros([IMAGE_X,IMAGE_Y])
        # self.orig[IMAGE_X//2,IMAGE_Y//2]=1
        # self.orig[1,1]=1
        self.step_cnt = 0

    def reset(self):
        #self.update()
        time.sleep(0.1)
        #self.canvas.delete(self.agentwin)
        self.init_scenario()
        s_ = np.array([1.0 * self.pos['x'] / IMAGE_X, 1.0 * self.pos['y'] / IMAGE_Y])
        return s_

    def step(self, action):
        self.reconstruction[self.pos['x'],self.pos['y']] = self.orig[self.pos['x'],self.pos['y']]
        # reward = self.orig[self.pos['x'],self.pos['y']]#todo -
        reward = -np.sqrt(np.mean((self.reconstruction - self.orig)**2))
        self.reward_list.append(reward)
        self.value = self.value_alpha*reward + (1-self.value_alpha)*self.value

        self.reconstruction *= RECONSTRUCT_DECAY

        if action == 0:   # up
                self.pos['y'] += 1 if self.pos['y'] < IMAGE_Y-1 else -REFLECT_EN
        elif action == 1:   # down
                self.pos['y'] -= 1 if self.pos['y'] > 0 else -REFLECT_EN
        elif action == 2:   # left
            self.pos['x'] -= 1 if self.pos['x'] > 0 else -REFLECT_EN
        elif action == 3:   # right
            self.pos['x'] += 1 if self.pos['x'] < IMAGE_X-1 else -REFLECT_EN

        self.step_cnt +=1

        # reward function
        if self.step_cnt % 5000 == 0:
            print('step_cnt= ', self.step_cnt ,'r=',reward, 'value = ', self.value)
            self.plot_reward()

        done = self.step_cnt > MAX_STEPS
        s_ = np.array([1.0*self.pos['x']/IMAGE_X, 1.0*self.pos['y']/IMAGE_Y])
        if self.step_cnt % 1 == 0 and self.graphics_on:

            self.ax.imshow(self.reconstruction, interpolation='none', cmap='gray', vmin=0, vmax=1)
            # self.rect = patches.Rectangle((self.pos['x'], self.pos['y']), 0.1, 0.1, linewidth=1, edgecolor='r', facecolor='none')
            # self.ax.add_patch(self.rect)
            # plt.imshow(self.reconstruction, interpolation='none', cmap='gray', vmin=0, vmax=1)
            #plt.title('')
            #plt.show(block=False)
            self.rect.set_x(self.pos['x'])
            self.rect.set_y(self.pos['y'])
            plt.draw()
            #self.ax.draw()
            plt.pause(0.05)
        #plt.show()
        return s_, reward, done
    def observation_space(self): #todo define more generally
        ob = np.zeros([IMAGE_X*IMAGE_Y,2])
        for x in range(IMAGE_X):
            for y in range(IMAGE_Y):
                ob[x*IMAGE_Y+y,:]=[np.double(x)/IMAGE_X,np.double(y)/IMAGE_Y]
        return ob
    
    def num2srt_actions(self,actions_by_num):
        return [self.action_space[a] for a in actions_by_num]

    def render(self):
        # time.sleep(0.01)
        self.update()

    def plot_reward(self):
        self.ax.clear()
        self.reward_plot_handler, =plt.plot(self.reward_list)
        plt.pause(0.05)

    def save_train_history(self):
        with open(self.history_file,'wb') as f:
            pickle.dump([self.reward_list,self.q_snapshots,self.orig],f)


