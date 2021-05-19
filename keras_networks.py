import tensorflow.keras as keras
def rnn_model_101(n_timesteps=5,lr=1e-3,dropout=0.0):
    inputA = keras.layers.Input(shape=(n_timesteps,28,28,1))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define CNN model
    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
    x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

    x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
    print(x1.shape)

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)
    # print(x1.shape)

    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(dropout))(x1)
    print(x1.shape)

    x = keras.layers.Concatenate()([x1,inputB])
    print(x.shape)
    # define LSTM model
    x = keras.layers.GRU(100,input_shape=(n_timesteps, None),return_sequences=False)(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x)
    opt=keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model