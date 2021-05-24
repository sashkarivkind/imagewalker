
	
# example of loading the cifar10 dataset
from matplotlib import pyplot
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY
 
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


#32,32,64,64,128,128
def net():
    input = keras.layers.Input(shape=(32,32,3))
    
    #Define CNN
    x = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(input)
    x = keras.layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(x)
    x = keras.layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')(x)
    x = keras.layers.Conv2D(128,(3,3),activation='relu', padding = 'same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.2)(x)
    #Flatten and add linear layer and softmax
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128,activation="relu")(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    
    model = keras.models.Model(inputs=input,outputs=x)
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
    

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = 'cifar_accuracy'
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()
    
# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = net()
    # fit model
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
 
# entry point, run the test harness
run_test_harness()