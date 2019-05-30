# coding=utf-8

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
import numpy as np

from src import models
from src import features


class DatasetGenerator(models.AbstractDatasetGenerator):
    @staticmethod
    def _get_one_features_part(file_path, **kwargs):
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

    def get_file_features(self, file_path, **kwargs):
        b1 = self._get_one_features_part(file_path, d=3, p=1, k=5, mfccd=True, mfccdd=True, remove_silence=True)
        b2 = self._get_one_features_part(file_path, d=1, p=1, k=7, mfccd=True, mfccdd=True, remove_silence=True)
        return np.vstack([b1, b2])


def get_model(classes=11):
    model = Sequential()

    # we can think of this chunk as the input layer
    model.add(Dense(200, input_dim=546, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # we can think of this chunk as the hidden layer
    model.add(Dense(128, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # we can think of this chunk as the hidden layer
    model.add(Dense(64, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(80, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=0.001),
                  metrics=['accuracy'])

    return model
