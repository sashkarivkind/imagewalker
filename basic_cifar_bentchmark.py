
	
# example of loading the cifar10 dataset
import numpy as np
from matplotlib import pyplot
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
images, labels = trainX, trainY

paramaters = {
    'epochs' : 10,
    
    'regularizer' : None,
    
    'optimizer' : tf.keras.optimizers.Adam,
    
    'dropout' : 0.4,

    'lr' : 1e-3,
    
    'run_id' : np.random.randint(1000,9000)
    }
   
print(paramaters)

epochs = paramaters['epochs']
regularizer = paramaters['regularizer']
optimizer = paramaters['optimizer']
dropout = paramaters['dropout']
lr = paramaters['lr']
run_id = paramaters['run_id']
# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)

#%%

path = '/home/orram/Documents/GitHub/imagewalker/teacher_student/'
# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

    

def net():
    input = keras.layers.Input(shape=(32,32,3))
    
    #Define CNN
    x = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn1')(input)
    x = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2), 
                            name = 'max_pool1')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn2')(x)
    x = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn22')(x)
    x = keras.layers.MaxPooling2D((2, 2), 
                            name = 'max_pool2')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn3')(x)
    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same', 
                            name = 'cnn32')(x)
    x = keras.layers.MaxPooling2D((2, 2), 
                            name = 'max_pool3')(x)
    x = keras.layers.Dropout(0.2)(x)
    #Flatten and add linear layer and softmax
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128,activation="relu", 
                            name = 'fc1')(x)
    x = keras.layers.Dense(10,activation="softmax", 
                            name = 'final')(x)
    
    model = keras.models.Model(inputs=input,outputs=x, name = 'base_cifar')
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


# define model
model = net()

history = model.fit(trainX, 
                    trainY, 
                    epochs=15, 
                    batch_size=64, 
                    validation_data=(testX, testY), 
                    verbose=1)

#Save Network
model.save(path +'cifar_trained_model')

#plot_results
plt.figure()
plt.plot(history.history['sparse_categorical_accuracy'], label = 'train')
plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'test')
plt.legend()
plt.grid()
plt.title('Cifar10 - train/test accuracies')
plt.savefig('Saved Networks accur plot')