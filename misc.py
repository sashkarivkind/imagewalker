import matplotlib.pyplot as plt
import numpy as np

class HP:
    pass

class Recorder:
    def __init__(self,n=2):
        self.n = n
        self.fig, self.ax = plt.subplots(n,3)
        self.records = []
        self.running_average_alpha = 1.0/1000.0
        self.running_averages = []
        for ii in range(self.n):
            self.records.append([])
            self.running_averages.append([0])

    def record(self,this_item):
        for ii in range(self.n):
            this_running_average = self.running_average_alpha*this_item[ii]+(1-self.running_average_alpha)*self.running_averages[ii][-1]
            self.records[ii].append(this_item[ii])
            self.running_averages[ii].append(this_running_average)
    def plot(self):
        for ii in range(self.n):
            self.ax[ii,0].clear()
            self.ax[ii,0].plot(self.records[ii])
            self.ax[ii,0].plot(self.running_averages[ii])
            self.ax[ii,1].clear()
            self.ax[ii,1].plot(self.records[ii][-1000:])
            self.ax[ii,2].clear()
            self.ax[ii,2].plot(np.abs(np.fft.fft(self.records[ii][-10000:]))[:1000])
        plt.pause(0.05)

def magnify_image(img,factor):
    return np.kron(img,np.ones([factor,factor]))

def image_from_np(img, scale=256,size_fac=1):
    return ImageTk.PhotoImage(image=Image.fromarray(scale*magnify_image(img,size_fac)))