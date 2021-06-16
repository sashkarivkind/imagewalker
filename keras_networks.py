import tensorflow.keras as keras
def rnn_model_101(n_timesteps=5,lr=1e-3,dropout=0.0,ignore_input_B=False,rnn_type='gru'):
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

    # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2s, 2)))(x1)
    # print(x1.shape)

    x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
    x1=keras.layers.TimeDistributed(keras.layers.Dropout(dropout))(x1)
    print(x1.shape)

    x = keras.layers.Concatenate()([x1,inputB]) if not ignore_input_B else x1
    print(x.shape)

    if rnn_type=='gru':
        x = keras.layers.GRU(100,input_shape=(n_timesteps, None),return_sequences=False)(x)
    elif rnn_type=='rnn':
        x = keras.layers.SimpleRNN(100,input_shape=(n_timesteps, None),return_sequences=False)(x)
    else:
        error

    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x)
    opt=keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def rnn_model_102(n_timesteps=5,lr=1e-3,dropout=0.0,ignore_input_B=False,rnn_type='gru',rnn_layers=1,input_size=(28,28,1),conv_fe=False):
    inputA = keras.layers.Input(shape=(n_timesteps,)+input_size)
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    if conv_fe:
    # define CNN model
        x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
        x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

        # x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
        # x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
        # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

        # x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
        # print(x1.shape)

        # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2s, 2)))(x1)
        # print(x1.shape)

        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.Dropout(dropout))(x1)
        print(x1.shape)
    else:
        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(inputA)

    x = keras.layers.Concatenate()([x1,inputB]) if not ignore_input_B else x1
    print(x.shape)

    for ll in range(rnn_layers):
        return_sequence = ll+1 < rnn_layers
        print('debu return sequence',return_sequence)
        if rnn_type=='gru':
            x = keras.layers.GRU(100,input_shape=(n_timesteps, None),return_sequences=return_sequence)(x)
        elif rnn_type=='rnn':
            x = keras.layers.SimpleRNN(100,input_shape=(n_timesteps, None),return_sequences=return_sequence)(x)
        else:
            error

    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x)
    opt=keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def rnn_model_103(n_timesteps=5,lr=1e-3,dropout=0.0,ignore_input_B=False,rnn_type='gru',input_size=(28,28,1),conv_fe=False, rnn_units=100):
    inputA = keras.layers.Input(shape=(n_timesteps,)+input_size)
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    rnn_state = keras.layers.Input(shape=(rnn_units))

    if conv_fe:
    # define CNN model
        x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
        x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

        x1=keras.layers.TimeDistributed(keras.layers.Conv2D(32,(3,3),activation='relu'))(x1)
        x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

        x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(x1)
        print(x1.shape)

        # x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2s, 2)))(x1)
        # print(x1.shape)

        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.Dropout(dropout))(x1)
        print(x1.shape)
    else:
        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(inputA)

    x = keras.layers.Concatenate()([x1,inputB]) if not ignore_input_B else x1
    print(x.shape)

    if rnn_type=='gru':
        x = keras.layers.GRU(100,input_shape=(n_timesteps, None),return_sequences=False)(x, initial_state=rnn_state)
    elif rnn_type=='rnn':
        x = keras.layers.SimpleRNN(100,input_shape=(n_timesteps, None),return_sequences=False)(x, initial_state=rnn_state)
    else:
        error
    rnn_state_out = x
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB,rnn_state],outputs=x)
    model_1_step = keras.models.Model(inputs=[inputA,inputB,rnn_state],outputs=[x,rnn_state_out])
    opt=keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    # model_1_step.compile()

    return model, model_1_step

def convlstm_v2(n_timesteps = 5, cell_size = 128, input_size = 28,input_dim = 1, concat = False):
    inputA = keras.layers.Input(shape=(n_timesteps,input_size,input_size,input_dim))
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    # define LSTM model
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(inputA)
    print(x.shape)
    # x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(x)
    # print(x.shape)
    x = keras.layers.ConvLSTM2D(cell_size, 2, dropout = 0.2, recurrent_dropout=0.1, return_sequences=True)(x)
    print(x.shape)
    x = keras.layers.Flatten()(x)
    print(x.shape)
    if concat:
        x = keras.layers.Concatenate()([x,inputB])
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'ConvLSTM_{}'.format(concat))
    opt=tf.keras.optimizers.Adam(lr=3e-3)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model