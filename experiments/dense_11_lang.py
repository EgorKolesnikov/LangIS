# coding=utf-8

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
import numpy as np

from src import models
from src import features


class DatasetGenerator(models.AbstractDatasetGenerator):
    def get_file_features(self, file_path, **kwargs):
        values, sr = features.read_wav(file_path)

        mfccs = features.mfcc(values, sr, **kwargs)
        sdcs_mfcc = features.sdc(mfccs[:13, :].T, **kwargs).T
        sdcs_d = features.sdc(mfccs[13:26, :].T, **kwargs).T
        sdcs_dd = features.sdc(mfccs[26:39, :].T, **kwargs).T
        res = np.vstack([mfccs, sdcs_mfcc, sdcs_d, sdcs_dd])

        # remove silence
        flags = features.get_columns_silence_flags(mfccs, **kwargs)
        res = res[:, flags]

        return res


def get_model(classes=11):
    model = Sequential()

    # we can think of this chunk as the input layer
    model.add(Dense(200, input_dim=234, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # we can think of this chunk as the hidden layer
    model.add(Dense(128, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # we can think of this chunk as the hidden layer
    model.add(Dense(80, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=0.0005),
                  metrics=['accuracy'])

    return model
