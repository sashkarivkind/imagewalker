import numpy as np
import tensorflow as tf
from tensorflow import keras

def baseline_net(*args):
    input = keras.layers.Input(shape=(32, 32, 3))

    # Define CNN
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='cnn1')(input)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool1')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn2')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn22')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool2')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            name='cnn3')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            name='cnn32')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool3')(x)
    x = keras.layers.Dropout(0.2)(x)
    # Flatten and add linear layer and softmax'''

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu",
                           name='fc1')(x)
    x = keras.layers.Dense(10, activation="softmax",
                           name='final')(x)

    model = keras.models.Model(inputs=input, outputs=x)
    # opt = tf.keras.optimizers.Adam(lr=1e-3)
    #
    # model.compile(
    #     optimizer=opt,
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["sparse_categorical_accuracy"],
    # )
    return model

def baseline_net_with_skips(*args):
    input = keras.layers.Input(shape=(32, 32, 3))

    # Define CNN
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='cnn1')(input)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool1')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn2')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn22')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool2')(x)
    skip = x
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            name='cnn3')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            name='cnn32')(x)

    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn33')(x)

    x= keras.layers.add([x,skip])
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool3')(x)
    x = keras.layers.Dropout(0.2)(x)
    # Flatten and add linear layer and softmax'''

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu",
                           name='fc1')(x)
    x = keras.layers.Dense(10, activation="softmax",
                           name='final')(x)

    model = keras.models.Model(inputs=input, outputs=x)
    # opt = tf.keras.optimizers.Adam(lr=1e-3)
    #
    # model.compile(
    #     optimizer=opt,
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["sparse_categorical_accuracy"],
    # )
    return model


def baseline_net_with_skips_deeper(*args):
    input = keras.layers.Input(shape=(32, 32, 3))

    # Define CNN
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='cnn1')(input)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool1')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn2')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn22')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool2')(x)
    for ii in range(3):
        skip = x
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                name='cnn3_'+str(ii))(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                name='cnn32_'+str(ii))(x)

        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                name='cnn33_'+str(ii))(x)

        x = keras.layers.add([x, skip])

    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool3')(x)
    x = keras.layers.Dropout(0.2)(x)
    # Flatten and add linear layer and softmax'''

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu",
                           name='fc1')(x)
    x = keras.layers.Dense(10, activation="softmax",
                           name='final')(x)

    model = keras.models.Model(inputs=input, outputs=x)
    # opt = tf.keras.optimizers.Adam(lr=1e-3)
    #
    # model.compile(
    #     optimizer=opt,
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["sparse_categorical_accuracy"],
    # )
    return model

def baseline_net_with_skips_deeper_laynorm(*args):
    input = keras.layers.Input(shape=(32, 32, 3))

    # Define CNN
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='cnn1')(input)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            name='cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool1')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn2')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            name='cnn22')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool2')(x)
    for ii in range(3):
        skip = x
        x = keras.layers.LayerNormalization(axis=3)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                name='cnn3_'+str(ii))(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                name='cnn32_'+str(ii))(x)

        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                name='cnn33_'+str(ii))(x)

        x = keras.layers.add([x, skip])

    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool3')(x)
    x = keras.layers.LayerNormalization(axis=3)(x)
    # Flatten and add linear layer and softmax'''

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu",
                           name='fc1')(x)
    x = keras.layers.Dense(10, activation="softmax",
                           name='final')(x)

    model = keras.models.Model(inputs=input, outputs=x)
    # opt = tf.keras.optimizers.Adam(lr=1e-3)
    #
    # model.compile(
    #     optimizer=opt,
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["sparse_categorical_accuracy"],
    # )
    return model


def parametric_net(dropout1=0.2, dropout2=0.0, resblocks=3, layer_norm_res=True, layer_norm_2=True, skip_conn=True, last_maxpool_en=True, nl='relu',last_layer_size=128):
    input = keras.layers.Input(shape=(32, 32, 3))

    # Define CNN
    x = keras.layers.Conv2D(32, (3, 3), activation=nl, padding='same',
                            name='cnn1')(input)
    x = keras.layers.Conv2D(32, (3, 3), activation=nl, padding='same',
                            name='cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool1')(x)
    x = keras.layers.Dropout(dropout1)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                            name='cnn2')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                            name='cnn22')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool2')(x)
    for ii in range(resblocks):
        skip = x
        if layer_norm_res:
            x = keras.layers.LayerNormalization(axis=3)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation=nl, padding='same',
                                name='cnn3_'+str(ii))(x)
        x = keras.layers.Conv2D(128, (3, 3), activation=nl, padding='same',
                                name='cnn32_'+str(ii))(x)

        x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                                name='cnn33_'+str(ii))(x)
        if skip_conn:
            x = keras.layers.add([x, skip])

    if last_maxpool_en:
        x = keras.layers.MaxPooling2D((2, 2),
                                      name='max_pool3')(x)
    else:
        x = keras.layers.MaxPooling2D((1, 1),
                                      name='max_pool3')(x) #todo: strange behavior if nothing here.. don't know why

    if layer_norm_2:
        x = keras.layers.LayerNormalization(axis=3)(x)
    # Flatten and add linear layer and softmax'''

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(last_layer_size, activation=nl,
                           name='fc1')(x)

    x = keras.layers.Dropout(dropout2)(x)

    x = keras.layers.Dense(10, activation="softmax",
                           name='final')(x)

    model = keras.models.Model(inputs=input, outputs=x)
    # opt = tf.keras.optimizers.Adam(lr=1e-3)
    #
    # model.compile(
    #     optimizer=opt,
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["sparse_categorical_accuracy"],
    # )
    return model


def parametric_net_befe(dropout1=0.2, dropout2=0.0, resblocks=3, layer_norm_res=True, layer_norm_2=True, skip_conn=True,
                        last_maxpool_en=True, nl='relu', last_layer_size=128, dense_interface=False):
    ### front end model
    input0 = keras.layers.Input(shape=(32, 32, 3))

    # Define CNN
    x = keras.layers.Conv2D(32, (3, 3), activation=nl, padding='same',
                            name='cnn1')(input0)
    x = keras.layers.Conv2D(32, (3, 3), activation=nl, padding='same',
                            name='cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool1')(x)
    x = keras.layers.Dropout(dropout1)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                            name='cnn2')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                            name='cnn22')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool2')(x)
    if dense_interface:
        x = keras.layers.Conv2D(64, (3, 3), padding='same',
                                name='anti_sparse')(x)
    if layer_norm_res:
        x = keras.layers.LayerNormalization(axis=3)(x)

    fe_model = keras.models.Model(inputs=input0, outputs=x, name='frontend')

    #######  backend model
    input1 = keras.layers.Input(shape=np.shape(x)[1:])
    x = input1
    for ii in range(resblocks):
        skip = x
        if layer_norm_res:
            x = keras.layers.LayerNormalization(axis=3)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation=nl, padding='same',
                                name='cnn3_' + str(ii))(x)
        x = keras.layers.Conv2D(128, (3, 3), activation=nl, padding='same',
                                name='cnn32_' + str(ii))(x)

        x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                                name='cnn33_' + str(ii))(x)
        if skip_conn:
            x = keras.layers.add([x, skip])

    if last_maxpool_en:
        x = keras.layers.MaxPooling2D((2, 2),
                                      name='max_pool3')(x)
    else:
        x = keras.layers.MaxPooling2D((1, 1),
                                      name='max_pool3')(x)  # todo: strange behavior if nothing here.. don't know why

    if layer_norm_2:
        x = keras.layers.LayerNormalization(axis=3)(x)
    # Flatten and add linear layer and softmax'''

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(last_layer_size, activation=nl,
                           name='fc1')(x)

    x = keras.layers.Dropout(dropout2)(x)

    x = keras.layers.Dense(10, activation="softmax",
                           name='final')(x)

    be_model = keras.models.Model(inputs=input1, outputs=x, name='backend')

    model = tf.keras.models.Sequential()
    model.add(fe_model)
    model.add(be_model)

    return model  # fe_model, be_model

def parametric_net_befe_v2(dropout1=0.2, dropout2=0.0, resblocks16=3,resblocks8=3, layer_norm_res16=True,layer_norm_res8=True, layer_norm_2=True, skip_conn16=True,skip_conn8=True,
                        dense_interface=True,last_maxpool_en=True, nl='relu', last_layer_size=128):
    ### front end model
    input0 = keras.layers.Input(shape=(32, 32, 3))

    # Define CNN
    x = keras.layers.Conv2D(32, (3, 3), activation=nl, padding='same',
                            name='cnn1')(input0)
    x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                            name='cnn12')(x)
    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool1')(x)
    x = keras.layers.Dropout(dropout1)(x)
    for ii in range(resblocks16):
        skip = x
        if layer_norm_res16:
            x = keras.layers.LayerNormalization(axis=3)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation=nl, padding='same',
                                name='cnn21_' + str(ii))(x)
        x = keras.layers.Conv2D(128, (3, 3), activation=nl, padding='same',
                                name='cnn22_' + str(ii))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                                name='cnn23_' + str(ii))(x)
        if skip_conn16:
            x = keras.layers.add([x, skip])

    x = keras.layers.MaxPooling2D((2, 2),
                                  name='max_pool2')(x)
    if dense_interface:
        x = keras.layers.Conv2D(64, (3, 3), padding='same',
                                name='anti_sparse')(x)

    if layer_norm_res8:
        x = keras.layers.LayerNormalization(axis=3)(x)

    fe_model = keras.models.Model(inputs=input0, outputs=x, name='frontend')

    #######  backend model
    input1 = keras.layers.Input(shape=np.shape(x)[1:])

    # if dense_interface:
    #     x = keras.layers.ELU()(x)

    x = input1
    for ii in range(resblocks8):
        skip = x
        if layer_norm_res8:
            x = keras.layers.LayerNormalization(axis=3)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation=nl, padding='same',
                                name='cnn3_' + str(ii))(x)
        x = keras.layers.Conv2D(128, (3, 3), activation=nl, padding='same',
                                name='cnn32_' + str(ii))(x)

        x = keras.layers.Conv2D(64, (3, 3), activation=nl, padding='same',
                                name='cnn33_' + str(ii))(x)
        if skip_conn8:
            x = keras.layers.add([x, skip])

    if last_maxpool_en:
        x = keras.layers.MaxPooling2D((2, 2),
                                      name='max_pool3')(x)
    else:
        x = keras.layers.MaxPooling2D((1, 1),
                                      name='max_pool3')(x)  # todo: strange behavior if nothing here.. don't know why

    if layer_norm_2:
        x = keras.layers.LayerNormalization(axis=3)(x)
    # Flatten and add linear layer and softmax'''

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(last_layer_size, activation=nl,
                           name='fc1')(x)

    x = keras.layers.Dropout(dropout2)(x)

    x = keras.layers.Dense(10, activation="softmax",
                           name='final')(x)

    be_model = keras.models.Model(inputs=input1, outputs=x, name='backend')

    model = tf.keras.models.Sequential()
    model.add(fe_model)
    model.add(be_model)

    return model  # fe_model, be_model