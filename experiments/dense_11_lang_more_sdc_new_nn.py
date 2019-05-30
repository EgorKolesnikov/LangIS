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

        mfccs_d_dd = features.mfcc(values, sr, mfccd=True, mfccdd=True)
        sdcs_mfcc = features.sdc(mfccs_d_dd[:13, :].T, d=1, p=3, k=7).T
        sdcs_d = features.sdc(mfccs_d_dd[13:26, :].T, d=1, p=3, k=7).T
        sdcs_dd = features.sdc(mfccs_d_dd[26:39, :].T, d=1, p=3, k=7).T

        sdcs_mfcc_2 = features.sdc(mfccs_d_dd[:13, :].T, d=5, p=2, k=5).T
        sdcs_mfcc_3 = features.sdc(mfccs_d_dd[:13, :].T, d=3, p=1, k=5).T
        sdcs_mfcc_4 = features.sdc(mfccs_d_dd[:13, :].T, d=1, p=2, k=6).T

        res = np.vstack([mfccs_d_dd, sdcs_mfcc, sdcs_d, sdcs_dd, sdcs_mfcc_2, sdcs_mfcc_3, sdcs_mfcc_4])

        # remove silence
        flags = features.get_columns_silence_flags(mfccs_d_dd, **kwargs)
        res = res[:, flags]
        return res


def get_model(classes=11):
    model = Sequential()

    # we can think of this chunk as the input layer
    model.add(Dense(520, input_dim=520))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # we can think of this chunk as the hidden layer
    model.add(Dense(256))
    model.add(Activation('relu'))

    # we can think of this chunk as the hidden layer
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=0.0009),
                  metrics=['accuracy'])

    return model
