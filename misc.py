import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from scipy import misc


class HP:
    pass

class Recorder:
    def __init__(self,n=2, do_fft = False):
        self.n = n
        self.do_fft = do_fft
        self.records = []
        self.running_average_alpha = 1.0/1000.0
        self.running_averages = []
        for ii in range(self.n):
            self.records.append([])
            self.running_averages.append([0])
        self.plotzero = True


    def record(self,this_item):
        for ii in range(self.n):
            this_running_average = self.running_average_alpha*this_item[ii]+(1-self.running_average_alpha)*self.running_averages[ii][-1]
            self.records[ii].append(this_item[ii])
            self.running_averages[ii].append(this_running_average)
    def plot(self):
        if self.plotzero:
            self.fig, self.ax = plt.subplots(self.n, 3 if self.do_fft else 2)
            self.plotzero = False
        for ii in range(self.n):
            self.ax[ii,0].clear()
            self.ax[ii,0].plot(self.records[ii])
            self.ax[ii,0].plot(self.running_averages[ii])
            self.ax[ii,1].clear()
            self.ax[ii,1].plot(self.records[ii][-1000:])
            if self.do_fft:
                self.ax[ii,2].clear()
                self.ax[ii,2].plot(np.abs(np.fft.fft(self.records[ii][-10000:]))[:1000])
        plt.pause(0.05)
    def to_pickle(self):
        return [self.records, self.running_averages]

    def save(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.to_pickle(), f)

    def load(self,filename):
        with open(filename, 'rb') as f:
            [self.records, self.running_averages] = pickle.load(f)
            self.n = len(self.records)



def magnify_image(img,factor):
    return np.kron(img,np.ones([factor,factor]))

def image_from_np(img, scale=256,size_fac=1):
    return ImageTk.PhotoImage(image=Image.fromarray(scale*magnify_image(img,size_fac)))

def mean_by_bins(x,y,xbins):
    means=[]
    for xmin,xmax in zip(xbins[:-1],xbins[1:]):
        qq = np.logical_and(np.array(x)<=xmax,  np.array(x)>=xmin)
        means.append(np.mean(np.array(y)[qq]))
    return means

def kernel_weights_prep(n,m,w, kernel = None):
    if kernel is None:
        kernel = lambda x,mu: np.exp(-(x-mu)**2./(2.*w**2))/np.sqrt(np.pi)/w
    wk=np.zeros([n,m])
    for mm in range(m):
        v = np.linspace(0,m-1,n)
        wk[:,mm] = kernel(v,mm)
    return wk

def build_edge_states(maxx=10,vmax=10):
    state_table = np.zeros([0,maxx+2])
    for q in range(maxx):
        for z in range(1,maxx-q):
            for v in range(-vmax,vmax+1):
                this_state = np.zeros([maxx+2])
                this_state[q:q+z] = 1
                this_state[maxx]=v
                state_table = np.concatenate([state_table,[this_state]])
    return state_table

def max_by_different_argmax(a,b,axis=0):
    cc = np.argmax(b,axis=axis)
    if axis==0:
        return a[cc,list(range(len(cc)))]
    elif axis==1:
        return a[list(range(len(cc))),cc]

def pwl_to_wave(pwl):
    w_prev=[0,pwl[0][1]]
    wave =[]
    for w_this in pwl:
        wave += list(np.linspace(w_prev[1],w_this[1],max(w_this[0]-w_prev[0]+1,0)))[1:]
        w_prev = w_this
    return wave

def read_images_from_path(path = None, filenames = None):
    if filenames is None:
        filenames = sorted(glob.glob(path))
    images=[]
    for image_path in filenames:
        images.append( misc.imread(image_path))
    return images