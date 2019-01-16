"""
Reinforcement learning maze example.
This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

"""
import numpy as np
import time
import sys

from PIL import Image, ImageTk

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width
IMAGE_X = 3 #image size (set according to MNIST)
IMAGE_Y = 3
WIN_X = 10 #agents window
WIN_Y = 10
DISP_SCALE = 50

RECONSTRUCT_DECAY=0.99

MAX_STEPS=10000

WIN_X_HLF = (WIN_X-1) // 2
WIN_Y_HLF = (WIN_Y-1) // 2

def magnify_image(img,factor):
    return np.kron(img,np.ones([factor,factor]))

def image_from_np(img, scale=256,size_fac=1):
    return ImageTk.PhotoImage(image=Image.fromarray(scale*magnify_image(img,size_fac)))

class Image_env1(tk.Tk, object):
    def __init__(self):
        super(Image_env1, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('image reconstructor')
        self.geometry('{0}x{1}'.format(IMAGE_X * DISP_SCALE, IMAGE_Y * DISP_SCALE))
        self.reconstruction = np.ones([IMAGE_X, IMAGE_Y])
        self.pos = {'x':IMAGE_X // 2,'y':IMAGE_Y // 2}
        self.init_scenario()

    def init_scenario(self):
        self.orig=np.ones([IMAGE_X,IMAGE_Y])
        self.canvas = tk.Canvas(bg='blue',
                           height=IMAGE_Y*DISP_SCALE,
                           width=IMAGE_X*DISP_SCALE)
        img = image_from_np(self.reconstruction,size_fac=DISP_SCALE)
        # img = ImageTk.PhotoImage(image=Image.fromarray(self.reconstruction))
        self.picture_visu = self.canvas.create_image(0, 0, anchor='nw' , image=img)
        # self.agentwin = self.canvas.create_rectangle(
        #     self.pos['x'] - WIN_X_HLF, self.pos['y'] - WIN_Y_HLF,
        #     self.pos['x'] + WIN_X_HLF, self.pos['y'] + WIN_Y_HLF,
        #     fill='red')


        # pack all
        # self.canvas.pack()
        #self.after(100)
        self.step_cnt = 0

    def reset(self):
        #self.update()
        time.sleep(0.1)
        #self.canvas.delete(self.agentwin)
        #self.init_scenario()
        s_ = np.array([1.0 * self.pos['x'] / IMAGE_X, 1.0 * self.pos['y'] / IMAGE_Y])
        return s_

    def step(self, action):
        #s = self.canvas.coords(self.rect)
        # base_action = np.array([0, 0])
        self.reconstruction[self.pos['x'],self.pos['y']] = self.orig[self.pos['x'],self.pos['y']]
        if action == 0:   # up
            if self.pos['y'] < IMAGE_Y-1:
                self.pos['y'] += 1
        elif action == 1:   # down
            if self.pos['y'] > 0:
                self.pos['y'] -= 1
        elif action == 2:   # right
            if self.pos['x'] < IMAGE_X-1:
                self.pos['x'] += 1
        elif action == 3:   # left
            if self.pos['x'] > 0:
                self.pos['x'] -= 1
        self.step_cnt +=1
        self.reconstruction *= RECONSTRUCT_DECAY
        #self.canvas.move(self.agentwin, self.pos['x']*DISP_SCALE, self.pos['y']*DISP_SCALE)  # move agent

        #next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        reward = -np.sqrt(np.mean((self.reconstruction - self.orig)**2))

        done = self.step_cnt > MAX_STEPS
        s_ = np.array([1.0*self.pos['x']/IMAGE_X, 1.0*self.pos['y']/IMAGE_Y])
        # img = ImageTk.PhotoImage(image=Image.fromarray(self.reconstruction))
        if self.step_cnt % 1000 == 0:
            img = image_from_np(self.reconstruction, size_fac=DISP_SCALE)
            img = ImageTk.PhotoImage(image=Image.fromarray(100*self.orig))
            #self.canvas.itemconfig(self.picture_visu, image=img)
            #self.update()
            print(self.reconstruction)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()


