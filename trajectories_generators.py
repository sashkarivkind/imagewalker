#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 14:44:29 2021

@author: orram
"""
import numpy as np
import matplotlib.pyplot as plt
import random


#%%
######################## Function ###############################################
def create_trajectory(starting_point, sample = 5, style = 'brownian', noise = 0.15):
    steps = []
    phi = np.random.randint(0.1,2*np.pi) #the angle in polar coordinates
    speed = 0.8#np.abs(0.5 + np.random.normal(0,0.5))         #Constant added to the radios
    r = 3
    name_list = ['const direction + noise','ZigZag','spiral', 'brownian','degenerate']
    speed_noise = speed * 0.2
    phi_noise = 0.05
    x, y = starting_point[1], starting_point[0]
    steps.append([y,x])
    phi_speed =  (1/8)*np.pi
    old_style = style
    for j in range(sample-1):
        style = old_style
        if style == 'mix':
            old_style = 'mix'
            style = random.sample(name_list, 1)
        if style == 'const_p_noise':
            r += speed + np.random.normal(-0.5,speed_noise)
            phi_noise = noise
            phi_speed = np.random.normal(0,phi_noise)
            phi += phi_speed
        elif style == 'ZigZag':
            r += speed + np.random.normal(-0.5,speed_noise)
            phi_noise = 0.005
            phi_speed *=  -1
            phi += phi_speed + np.random.normal(0,phi_noise)
        elif style == 'spiral':
            r += speed/2 + np.random.normal(-0.5,speed_noise)
            phi_noise = 0.1
            phi_speed = np.random.normal((2/4)*np.pi,(1/8)*np.pi)
            factor = 1#np.random.choice([-1,1])
            phi += factor * phi_speed
        elif style == 'big_steps':
            r += speed/2 + np.random.normal(-0.5,speed_noise)
            phi_noise = 0.1
            phi_speed = np.random.normal((2/4)*np.pi,(1/8)*np.pi)
            factor = np.random.choice([-1,1])
            phi += factor * phi_speed
        elif style == 'brownian':
            r += speed/2 + np.random.normal(-0.5,speed_noise)
            phi = np.random.randint(0.1,2*np.pi)
        elif style == 'degenerate':
            r += speed + np.random.normal(-0.5,speed_noise)
        elif style == 'old':
            
            starting_point += np.random.randint(-2,3,2) 
            r = 0
            phi = 0
        x, y = starting_point[1] + int(r * np.cos(phi)), starting_point[0]+int(r * np.sin(phi))
        steps.append([y,x])
        
            
    return steps


#%% 
names = ['const_p_noise','ZigZag','spiral', 'brownian','degenerate']
starting_point = np.array([64//6,64//6])
canvas = np.zeros([64,64])

for i in range(5): 
       steps = create_trajectory(starting_point = starting_point, sample = 10,
                                 style = 'const_p_noise', noise = 1)
       steps = np.array(steps)
       val = 1
       for step in range(10):
           canvas[steps[step][0], steps[step][1]] = val
           val += 1
       plt.plot(steps[:,1],steps[:,0], label = i)
       starting_point += np.array([64//6,64//6])

plt.imshow(canvas)
plt.legend()
#%%
style_name = 'const_p_noise'
starting_point = np.array([64//2,64//2])
canvas = np.zeros([64,64])
fig, ax = plt.subplots(2)
for i in range(5): 
       steps = create_trajectory(starting_point = starting_point, sample = 10,
                                 style = style_name, noise = 0.5*i)
       steps = np.array(steps)
       val = 1
       for step in range(10):
           canvas[steps[step][0], steps[step][1]] = val
           val += 1
       #plt.plot(steps[:,1],steps[:,0], label = i)
       ax[0].plot(steps[:,1], label = i*0.5)
       ax[1].plot(steps[:,0], label = i*0.5)
plt.title(style_name)
plt.legend()


