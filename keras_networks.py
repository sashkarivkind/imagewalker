import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


class LossClass(keras.layers.Layer):

  def __init__(self):
      super(LossClass, self).__init__()
      # Create a non-trainable weight.
      self.loss = tf.Variable(initial_value=0)

  def call(self, y_true,y_pred):
      self.loss.assign(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred), axis=0))
      return self.total


def time_distributed_xentropy_loss(y_true,y_pred):

#     print('debug_shapes:',y_true.shape, y_true.shape,)
#     tt = keras.layers.TimeDistributed(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.log(y_pred),labels=y_true))
#     tt = tf.reduce_mean(tt, axis=1) #taking mean along the time dimension
#     tt = tf.reduce_mean(tt, axis=0) #taking mean along the batch dimension

    # y_true = K.reshape(y_true, (K.shape(y_true)[0], -1))
    y_true = keras.layers.Reshape((5,))(y_true)
    y_true = tf.cast(y_true, tf.int32)
    tt= tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=tf.log(y_pred))
    tt=tf.reduce_mean(tt)
    return tt

def time_distributed_accuracy(y_true,y_pred):
    # y_true = K.reshape(y_true, (K.shape(y_true)[0], -1))
    y_true = keras.layers.Reshape((5,))(y_true)
    y_true = tf.cast(y_true, tf.int64)
    print('debug_shapes:', np.shape(y_pred))
    print('debug_shapes:', np.shape(tf.argmax(y_pred,axis=-1)))
    tt= tf.equal(tf.argmax(y_pred,axis=-1), y_true)
    tt=tf.reduce_mean(tf.cast(tt,"float"))
    return tt

def time_distributed_accuracy_last_step(y_true,y_pred):
    # y_true = K.reshape(y_true, (K.shape(y_true)[0], -1))
    y_true = keras.layers.Reshape((5,))(y_true)
    y_true = tf.cast(y_true, tf.int64)
    print('debug_shapes:', np.shape(y_pred))
    print('debug_shapes:', np.shape(tf.argmax(y_pred,axis=-1)))
    tt= tf.equal(tf.argmax(y_pred,axis=-1), y_true)
    tt=tf.reduce_mean(tf.cast(tt,"float")[...,-1])
    return tt

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

def rnn_model_102(n_timesteps=5,lr=1e-3,dropout=0.0,ignore_input_B=False,rnn_type='gru',rnn_layers=1,input_size=(28,28,1),conv_fe=False, rnn_units = 100):

    inputA = keras.layers.Input(shape=(n_timesteps,)+input_size)
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    if conv_fe:
    # define CNN model
        x1=keras.layers.TimeDistributed(keras.layers.Conv2D(8,(3,3),activation='relu'))(inputA)
#         x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.Dropout(dropout))(x1)
        print(x1.shape)
    else:
        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(inputA)

    x = keras.layers.Concatenate()([x1,inputB]) if not ignore_input_B else x1
    print(x.shape)

    for ll in range(rnn_layers):
        return_sequence = ll+1 < rnn_layers
#         return_sequence = True
        print('debu return sequence',return_sequence)
        if rnn_type=='gru':
            x = keras.layers.GRU(rnn_units,input_shape=(n_timesteps, None),return_sequences=return_sequence)(x)
        elif rnn_type == 'rnn':
            x = keras.layers.SimpleRNN(rnn_units, input_shape=(n_timesteps, None), return_sequences=return_sequence)(x)
        else:
            error

    x = keras.layers.Dropout(dropout)(x)
#     x = keras.layers.TimeDistributed(keras.layers.Dense(10,activation="softmax"))(x)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x)
    opt=keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
#         loss=time_distributed_xentropy_loss,
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

def core_a1(inputA, inputB, n_timesteps=5,lr=1e-3,dropout=0.0,ignore_input_B=True,rnn_type='gru',rnn_layers=1,input_size=(28,28,1),conv_fe=False,rnn_units=100,return_last_sequence=False, **kwargs):
    if conv_fe:
    # define CNN model
        x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
        x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.Dropout(dropout))(x1)
        print(x1.shape)
    else:
        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(inputA)

    x = keras.layers.Concatenate()([x1,inputB]) if not ignore_input_B else x1
    print(x.shape)

    for ll in range(rnn_layers):
        return_sequence = ll+1 < rnn_layers or return_last_sequence
        print('debu return sequence',return_sequence)
        if rnn_type=='gru':
            x = keras.layers.GRU(rnn_units,input_shape=(n_timesteps, None),return_sequences=return_sequence)(x)
        elif rnn_type == 'rnn':
            x = keras.layers.SimpleRNN(rnn_units, input_shape=(n_timesteps, None), return_sequences=return_sequence)(x)
        else:
            error
    return x

def rnn_model_multicore_201(n_cores=5,n_timesteps=5,lr=1e-3,input_size=(28,28,1),**kwargs):
    kwargs['n_timesteps'] = n_timesteps #needs to be passed further
    kwargs['input_size'] = input_size #needs to be passed further
    kwargs['lr'] = lr #needs to be passed further
    inputA = keras.layers.Input(shape=(n_timesteps,)+input_size)
    inputB = keras.layers.Input(shape=(n_timesteps,2))
    x_cores = [core_a1(inputA, inputB, **kwargs) for uu in range(n_cores)]
    x = keras.layers.Concatenate()(x_cores)
    x = keras.layers.Dense(10,activation="softmax")(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x)
    opt=keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def rnn_model_multicore_202(n_cores=5,n_timesteps=5,lr=1e-3,input_size=(28,28,1),**kwargs):
    kwargs['n_timesteps'] = n_timesteps #needs to be passed further
    kwargs['input_size'] = input_size #needs to be passed further
    kwargs['lr'] = lr #needs to be passed further
    kwargs['ignore_input_B'] = True #needs to be passed further
    kwargs['return_last_sequence'] = True #needs to be passed further

    inputA = keras.layers.Input(shape=(n_timesteps,)+input_size)
    inputB = keras.layers.Input(shape=(n_timesteps,2))
 ## data path
    x_cores = [core_a1(inputA,inputB, **kwargs) for uu in range(n_cores)]
    print('single data core shape',x_cores[0].shape)

 ##attention path
    a = keras.layers.GRU(100, input_shape=(n_timesteps, None), return_sequences=True)(inputB)
    a = keras.layers.TimeDistributed(keras.layers.Dense(n_cores,activation="softmax"))(a)
    print('attention shape',a.shape)

 ## apply attention
    x_cores = tf.stack(x_cores)
    x_cores = tf.transpose(x_cores,[1,2,0,3])
    print('data  shape',x_cores.shape)

    # x = tf.matmul(a,x_cores)
    x = a[..., tf.newaxis]*x_cores
    x = tf.reduce_sum(x,axis=2)
    print('data times attention shape',x.shape)

## do one more GRU layer
    x = keras.layers.GRU(100, input_shape=(n_timesteps, None), return_sequences=False)(x)

    x = keras.layers.Dense(10,activation="softmax")(x)

    model = keras.models.Model(inputs=[inputA,inputB],outputs=x)
    opt=keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


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

def rnn_model_102e(n_timesteps=5,lr=1e-3,dropout=0.0,ignore_input_B=False,rnn_type='gru',rnn_layers=1,input_size=(28,28,1),conv_fe=False,rnn_units=100, **kwargs):
    inputA = keras.layers.Input(shape=(n_timesteps,)+input_size)
    inputB = keras.layers.Input(shape=(n_timesteps,2))

    if conv_fe:
    # define CNN model
        x1=keras.layers.TimeDistributed(keras.layers.Conv2D(16,(3,3),activation='relu'))(inputA)
        x1=keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(x1)

        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(x1)
        x1=keras.layers.TimeDistributed(keras.layers.Dropout(dropout))(x1)
        print(x1.shape)
    else:
        x1=keras.layers.TimeDistributed(keras.layers.Flatten())(inputA)

    x = keras.layers.Concatenate()([x1,inputB]) if not ignore_input_B else x1
    print(x.shape)

    for ll in range(rnn_layers):
        return_sequence = True #ll+1 < rnn_layers
        print('debu return sequence',return_sequence)
        if rnn_type=='gru':
            x = keras.layers.GRU(rnn_units,input_shape=(n_timesteps, None),return_sequences=return_sequence)(x)
        elif rnn_type == 'rnn':
            x = keras.layers.SimpleRNN(rnn_units, input_shape=(n_timesteps, None), return_sequences=return_sequence)(x)
        else:
            error

    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.TimeDistributed(keras.layers.Dense(10,activation="softmax"))(x)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x)
    opt=keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opt,
        loss=time_distributed_xentropy_loss,
        metrics=[time_distributed_accuracy, time_distributed_accuracy_last_step],
    )
    return model