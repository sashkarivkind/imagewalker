import numpy as np
from misc import HP


class ESN:
    def __init__(self,n_inputs=1):
        self.hp = HP()
        self.hp.alpha_force = 1.
        self.hp.N = 1000
        self.hp.g = 1.5
        self.hp.nl = np.tanh
        self.hp.dt=0.1
        self.hp.n_inputs = n_inputs
        self.reset()
    def reset(self):
        self.W=self.hp.g*np.random.normal(size=[self.hp.N,self.hp.N])/np.sqrt(self.hp.N)
        self.wfb=np.random.normal(size=[self.hp.N,1])
        self.win=np.random.normal(size=[self.hp.N,self.hp.n_inputs])
        self.wout=np.zeros([self.hp.N,1])
        self.x = np.random.normal(size=[self.hp.N,1])
        self.r = self.hp.nl(self.x)
        self.FORCE_reset()
    def step(self,f=None, uin=None):
        self.r = self.hp.nl(self.x)
        self.z =np.matmul(self.wout.transpose(),self.r)
        self.x *= 1-self.hp.dt
        self.x += self.hp.dt*(np.matmul(self.W,self.r) + self.wfb*(self.z if f is None else f) + (np.matmul(self.win,uin.reshape([-1,1])) if uin is not None else 0))
    def train_batch():
        error
    def train_FORCE(self, f_vec=None, uin_vec=None,tmax=None):
        rec = []
        self.FORCE_reset()
        for ti, (u,f) in enumerate(zip(uin_vec,f_vec)):
            self.step(uin=u)
            self.FORCE_step(f)
            rec.append(self.z)
        return rec
    def FORCE_reset(self):
            self.P = self.hp.alpha_force*np.eye(self.hp.N,self.hp.N)
    def FORCE_step(self,f):
            k = np.matmul(self.P,self.r)
            rPr = np.matmul(self.r.transpose(),k)
            c = 1.0/(1.0 + rPr)
            self.P -= np.matmul(k,np.transpose(k*c))
            e = self.z-f
            dw = -e*k*c
            self.wout += dw