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
        mfccs = features.mfcc(values, sr, mfccd=True, mfccdd=True)

        sdcs_mfcc_1 = features.sdc(mfccs[:13, :].T, d=1, p=3, k=7).T
        sdcs_d_1 = features.sdc(mfccs[13:26, :].T, d=1, p=3, k=7).T
        sdcs_dd_1 = features.sdc(mfccs[26:39, :].T, d=1, p=3, k=7).T

        sdcs_mfcc_2 = features.sdc(mfccs[:13, :].T, d=3, p=1, k=5).T
        sdcs_d_2 = features.sdc(mfccs[13:26, :].T, d=3, p=1, k=5).T
        sdcs_dd_2 = features.sdc(mfccs[26:39, :].T, d=3, p=1, k=5).T

        sdcs_mfcc_3 = features.sdc(mfccs[:13, :].T, d=1, p=1, k=7).T
        sdcs_d_3 = features.sdc(mfccs[13:26, :].T, d=1, p=1, k=7).T
        sdcs_dd_3 = features.sdc(mfccs[26:39, :].T, d=1, p=1, k=7).T

        res = np.vstack(
            [mfccs, sdcs_mfcc_1, sdcs_d_1, sdcs_dd_1, sdcs_mfcc_2, sdcs_d_2, sdcs_dd_2, sdcs_mfcc_3, sdcs_d_3,
             sdcs_dd_3])

        # remove silence
        flags = features.get_columns_silence_flags(mfccs, **kwargs)
        res = res[:, flags]

        return res


def get_model(classes=11):
    model = Sequential()

    # we can think of this chunk as the input layer
    model.add(Dense(600, input_dim=780, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # we can think of this chunk as the hidden layer
    model.add(Dense(512, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # we can think of this chunk as the hidden layer
    model.add(Dense(256, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=0.0009),
                  metrics=['accuracy'])

    return model
