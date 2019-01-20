
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
UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width

IMAGE_NUM=4

IMAGE_X = 28 #image size (set according to MNIST)
IMAGE_Y = 28
WIN_X =  10 #agents window
WIN_Y = 10
DISP_SCALE = 50
REFLECT_EN = 0
RECONSTRUCT_DECAY=0.9

MAX_STEPS=3000

WIN_X_HLF = (WIN_X) // 2
WIN_Y_HLF = (WIN_Y) // 2

def magnify_image(img,factor):
    return np.kron(img,np.ones([factor,factor]))

def image_from_np(img, scale=256,size_fac=1):
    return ImageTk.PhotoImage(image=Image.fromarray(scale*magnify_image(img,size_fac)))

class Image_env1( object):
    def __init__(self,bmp_features = False):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.graphics_on=False
        # self.title('image reconstructor')
        # self.geometry('{0}x{1}'.format(IMAGE_X * DISP_SCALE, IMAGE_Y * DISP_SCALE))
        self.reconstruction = np.zeros([IMAGE_Y, IMAGE_X])
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

        #initialize graphics if relevant
        self.fig,self.ax = plt.subplots(1) #=plt.figure()
        if self.graphics_on:
            self.rect = patches.Rectangle((self.pos['x'], self.pos['y']), 0.1, 0.1, linewidth=1, edgecolor='r',
                                          facecolor='none')
            self.ax.add_patch(self.rect)
        plt.ion()
        plt.pause(0.1)
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

    def step(self, action):
        self.copy_current_region()
        reward = -np.sqrt(np.mean((self.reconstruction - self.orig)**2))
        self.reward_list.append(reward)
        self.value = self.value_alpha*reward + (1-self.value_alpha)*self.value
        self.value_list.append(self.value)
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
        # if self.step_cnt % 5000 == 0:
        #     print('step_cnt= ', self.step_cnt ,'r=',reward, 'value = ', self.value)
        #     self.plot_reward()

        done = self.step_cnt > MAX_STEPS
        s_ = self.current_features()
        if self.step_cnt % 1 == 0 and self.graphics_on:
            self.ax.imshow(self.reconstruction, interpolation='none', cmap='gray', vmin=0, vmax=1)
            self.rect.set_x(self.pos['x'])
            self.rect.set_y(IMAGE_Y-self.pos['y'])
            plt.draw()
            #self.ax.draw()
            plt.pause(0.05)
        #plt.show()
        return s_, reward, done
    def observation_space(self): #todo define more generally
        if not self.bmp_features:
            ob = np.zeros([IMAGE_X*IMAGE_Y,2])
            for x in range(IMAGE_X):
                for y in range(IMAGE_Y):
                    ob[x*IMAGE_Y+y,:]=[np.double(x)/IMAGE_X,np.double(y)/IMAGE_Y]
            return ob
        else:
            error('not supported yet')
    
    def num2srt_actions(self,actions_by_num):
        return [self.action_space[a] for a in actions_by_num]

    def render(self):
        # time.sleep(0.01)
        self.update()

    def plot_reward(self):
        self.ax.clear()
        self.reward_plot_handler, =plt.plot(self.reward_list)
        self.reward_plot_handler, =plt.plot(self.value_list)
        plt.pause(0.05)

    def save_train_history(self):
        with open(self.history_file,'wb') as f:
            pickle.dump([self.reward_list,self.q_snapshots,self.orig],f)

    def copy_current_region(self):
        max_x = np.min([self.pos['x']+WIN_X_HLF,IMAGE_X])
        max_y = np.min([self.pos['y']+WIN_Y_HLF,IMAGE_Y])
        min_x = np.max([self.pos['x']-WIN_X_HLF,0])
        min_y = np.max([self.pos['y']-WIN_Y_HLF,0])
        #approaching matrix coordinates with y first and y counted from above requires some care
        self.reconstruction[IMAGE_Y-max_y:IMAGE_Y-min_y,min_x:max_x] = self.orig[IMAGE_Y-max_y:IMAGE_Y-min_y,min_x:max_x]

    def current_region_flatten(self):
        max_x = self.pos['x']+WIN_X_HLF
        max_y = self.pos['y']+WIN_Y_HLF
        min_x = self.pos['x']-WIN_X_HLF
        min_y = self.pos['y']-WIN_Y_HLF
        #approaching matrix coordinates with y first and y counted from above requires some care
        return np.reshape(self.padded_orig[IMAGE_Y+WIN_Y_HLF-max_y:IMAGE_Y+WIN_Y_HLF-min_y
                          ,WIN_X_HLF+min_x:WIN_X_HLF+max_x], [-1])

    def current_features(self):
        if self.bmp_features:
            s_ = self.current_region_flatten()
        else:
            s_ = np.array([1.0*self.pos['x']/IMAGE_X, 1.0*self.pos['y']/IMAGE_Y])
        return s_


