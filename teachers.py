import numpy as np
from misc import *

class  Clipped_Harmonic_1D:
    def __init__(self):
        self.hp = HP()
        self.hp.name='Clipped_Harmonic_1D'
        self.hp.dt = 0.1
        self.hp.omega0 = 2*np.pi / 10.
        self.hp.A = 5.0 #target amplitude of oscillations. Target max speed is hence omega*A
        self.hp.noi = 0.01
        self.speed_lim = self.hp.omega0 * self.hp.A
        self.vel = 0
    def step(self,dvs_strip):
        dvs_strip = dvs_strip + self.hp.noi*np.random.normal(size=np.shape(dvs_strip))
        I = dvs_strip**2.0
        I = I/(np.sum(I)+1e-10)
        centre = np.sum(I*np.array(range(len(dvs_strip))))- len(dvs_strip)/2.0
#         print('centre:',centre)
        self.vel += self.hp.dt*self.hp.omega0*centre
        self.vel = self.vel if np.abs(self.vel)<self.speed_lim else self.speed_lim*np.sign(self.vel)

class  Clipped_Unstable_Harmonic_1D:
    def __init__(self):
        self.hp = HP()
        self.hp.name = 'Clipped_Unstable_Harmonic_1D'
        self.hp.dt = 0.1

        self.hp.omega0 = 2*np.pi / 10.
        self.hp.A = 5.0 #target amplitude of oscillations. Target max speed is hence omega*A
        self.hp.noi = 0.01
        self.speed_lim = self.hp.omega0 * self.hp.A
        self.vel = 0
        self.hp.gamma = 0.1
    def step(self,dvs_strip):
        dvs_strip = dvs_strip + self.hp.noi*np.random.normal(size=np.shape(dvs_strip))
        I = dvs_strip**2.0
        I = I/(np.sum(I)+1e-10)
        centre = np.sum(I*np.array(range(len(dvs_strip))))- len(dvs_strip)/2.0
#         print('centre:',centre)
        self.vel += self.hp.dt*self.hp.omega0*centre + self.hp.gamma * self.vel
        self.vel = self.vel if np.abs(self.vel)<self.speed_lim else self.speed_lim*np.sign(self.vel)