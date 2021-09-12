import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from scipy import misc
# from mnist import MNIST
import cv2
import sys
import copy


class HP:
    def update_attributes_from_file(self,file,attributes):
        with open(file,'rb') as f:
            temp_hp=pickle.load(f)
        for att in attributes:
           self.__dict__[att] = copy.copy(temp_hp.__dict__[att])
    def upadte_with_defaults(self,att={},default_att={}):
        for kk in default_att.keys():
            self.__dict__[kk] = att[kk] if kk in att.keys() else default_att[kk]
    def upadte_from_dict(self,att={}):
        for kk in att.keys():
            self.__dict__[kk] = att[kk]

class Logger:
    def __init__(self,log_name):
        self.log_name = log_name
        self.terminal = sys.stdout
        self.log = open(log_name, "w")
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.log_name, "a")
        self.log.write(message)
        self.log.close() #to make log readable at all times...

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
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


def one_hot(a,depth):
    o=np.zeros([len(a),depth])
    o[list(range(len(a))),a]=1
    return o

def softmax(x,axis=1):
    ee=np.exp(x)
    return(ee/np.sum(ee,axis=axis)[...,np.newaxis])

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



def read_images_from_path(path = None, filenames = None, max_image=1e7):
    if filenames is None:
        filenames = sorted(glob.glob(path))
    images=[]
    for cnt, image_path in enumerate(filenames):
        if cnt<max_image:
            images.append(1.0* plt.imread(image_path))#todo: doublecheck the scaling (may be x255 needed)
        else:
            break
    return images


def read_random_image_from_path(path = None, grayscale=False, padding=None,resize=None):
    filenames = sorted(glob.glob(path))
    filename=filenames[np.random.randint(len(filenames))]
    image=1.0* plt.imread(filename)#todo: doublecheck the scaling (may be x255 needed)
    if grayscale:
        if len(image.shape)==3:
            image=np.mean(image,axis=2)
    if resize is not None:
        image = cv2.resize(image,(int(image.shape[1]*resize),int(image.shape[0]*resize)))
    if padding is not None:
        image = add_padding(image,padding)
    return image,filename

def add_padding(image,padding):
    shp = np.array(image.shape)
    new_shp = shp + 0
    new_shp[:2] = new_shp[:2] + 2 * np.array(padding)
    new_image = np.zeros(new_shp)
    new_image[padding[0]:padding[0] + shp[0], padding[1]:padding[1] + shp[1]] = image
    return new_image

def relu_up_and_down(x,downsample_fun = lambda x: x):
    x=downsample_fun(x)
    up_down_views = [np.maximum(x,0), -np.minimum(x,0)]
    return np.concatenate([downsample_fun(uu).reshape([-1]) for uu in up_down_views])

def some_resized_mnist(size=(256,256), n=100,path='/home/bnapp/datasets/mnist/'):
    mnist = MNIST(path)
    images, labels = mnist.load_training()
    some_mnist =[ cv2.resize(0.0+np.reshape(uu,[28,28]), dsize=size) for uu in images[:n]]
    return some_mnist

def pack_scene(images,xys,xx=256,yy=256,y_size=28,x_size=28):
    #todo: double-check x-y vs. row-column convention
    scene=np.zeros([yy,xx])
    for image,xy in zip(images,xys):
        x0,y0=xy
        scene[y0:y0+y_size,x0:x0+x_size]=np.reshape(image,[y_size,x_size])
    # some_mnistSM =[ cv2.resize(1.+np.reshape(uu,[28,28]), dsize=(256, 256)) for uu in images[:20]]
    return scene

def build_mnist_scene(image_db,images_per_scene=3,xx=256,yy=256,y_size=28,x_size=28):
    #creates a canvas randomly filled by mnist digits
    #todo: double-check x-y vs. row-column convention
    images = [image_db[np.random.randint(len(image_db))] for uu in range(images_per_scene)]
    xys = [(np.random.randint(xx-x_size),np.random.randint(yy-y_size)) for uu in range(images_per_scene)]
    return pack_scene(images,xys,xx=xx,yy=yy,y_size=y_size,x_size=y_size)

def build_mnist_padded(image,xx=128,yy=128,y_size=28,x_size=28,offset=(0,0)):
    #todo: double-check x-y vs. row-column convention
    #prepares an mnist image padded with zeros everywhere around it, written in a somewhat strange way to resuse other availiable functions
    xys = [((xx-x_size)//2+offset[0],(yy-y_size)//2+offset[1])]
    return pack_scene(image,xys,xx=xx,yy=yy,y_size=y_size,x_size=y_size)

def build_cifar_padded(image,pad_size = 100, xx=132,yy=132,y_size=32,x_size=32,offset=(0,0)):
    #todo: double-check x-y vs. row-column convention
    #prepares an mnist image padded with zeros everywhere around it, written in a somewhat strange way to resuse other availiable functions
    
    image = cv2.copyMakeBorder( image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
    return image

def prep_mnist_sparse_images(max_image,images_per_scene=5,path='/home/bnapp/datasets/mnist/'):
    #todo: ensure not taking the same image over and over again
    mnist = MNIST(path)
    images, labels = mnist.load_training()
    return [build_mnist_scene(images,images_per_scene=images_per_scene) for uu in range(max_image)]

def prep_mnist_padded_images(max_image,size=None,path='/home/bnapp/datasets/mnist/'):
    if size is None: #kept separate from sale 1.0 to ensure backward compatibility
        mnist = MNIST(path)
        images, labels = mnist.load_training()
        return [build_mnist_padded([image]) for image in images[:max_image]]
    else:
        images=some_resized_mnist(size=size, n=max_image)
        return [build_mnist_padded([image],y_size=size[1],x_size=size[0]) for image in images[:max_image]]

def prep_n_grams(x,n=None,offsets=None):
    if (n is None) and  not (offsets is None):
        pass
    elif not(n is None) and (offsets is None):
        offsets = list(range(n))
    else:
        error('need to provide either n or offsets')
    ngram_dict = {}
    for ii in range(len(x)-offsets[-1]):
        this_ngram = tuple(x[ii+oo] for oo in offsets)
        if this_ngram in ngram_dict.keys():
            ngram_dict[this_ngram] +=1
        else:
            ngram_dict[this_ngram] = 1
    return ngram_dict

def undistort_q_poly(dq,w,cm=None, epsilon=1e-9): #undistort from fisheye, using polinomial fit
    if cm is None:
        cm = np.array([[0,0]])
    xy = dq-cm
    r = np.sqrt(np.sum(xy**2,axis=1))
    powvec = np.array([list(range(len(w)))])
    rnew = (r[..., np.newaxis]**powvec) @ w
    xynew = xy *(rnew / (r+epsilon))[..., np.newaxis]
    dqnew = xynew + cm
    return dqnew

def cifar_shape_fun(x,grayscale=False,normalize=True):
    o=x.reshape([-1,3,32,32]).transpose([0,2,3,1])
    if grayscale:
        o = o.mean(axis=3)[...,np.newaxis]
    if normalize:
        o = o/256.
    return o


