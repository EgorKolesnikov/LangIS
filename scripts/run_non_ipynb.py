import numpy as np
import IPython.display
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import shutil
from collections import defaultdict
from random import shuffle
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import np_utils
import pickle
from collections import Counter
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
from keras.layers.noise import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn import metrics
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import sklearn
from sklearn import mixture
import itertools




def _read_wav(file_path):
    try:
        sr, d = wavfile.read(file_path)

        if not str(d.dtype).startswith('float'):
            if d.dtype == 'int16':
                nb_bits = 16
            elif d.dtype == 'int32':
                nb_bits = 32
            else:
                print d.dtype, d
                raise Exception('???')

            max_nb_bit = float(2 ** (nb_bits - 1))
            d = d / (max_nb_bit + 1.0)

        if len(d.shape) == 2:
            d = d[:, 0]

        return d, sr
    except:
        # print traceback.format_exc()
        y, sr = librosa.load(file_path, sr=None)
        return y, sr


def _get_wav_mfcc(values, sr, winlen=25, winstep=15, n_mels=128, n_mfcc=13, mfccd=False, mfccdd=False, norm_mfcc=False, fmin=0, fmax=6000, **kwargs):
    winlen = int((winlen / 1000.0) * sr)
    winstep = int((winstep / 1000.0) * sr)

    mfcc = librosa.feature.mfcc(
        y=values,
        sr=sr,
        n_fft=winlen,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        hop_length=winstep,
        fmin=fmin,
        fmax=fmax
    )
    
    if norm_mfcc:
        mfcc = (mfcc - mfcc.mean()) / mfcc.var()

    blocks = [mfcc]

    if mfccd:
        d = librosa.feature.delta(mfcc)
        blocks.append(d)
    if mfccdd:
        dd = librosa.feature.delta(mfcc, order=2)
        blocks.append(dd)

    return np.vstack(blocks)


# Computing SDC
# From src:
# http://webcache.googleusercontent.com/search?q=cache:y_zQF8CnrysJ:www-lium.univ-lemans.fr/sidekit/_modules/frontend/features.html+&cd=2&hl=ru&ct=clnk&gl=ru&lr=lang_be%7Clang_ru&client=ubuntu

def _compute_delta(features, win=3, **kwargs):
    x = np.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=np.float32)
    x[:win, :] = features[0, :]
    x[win:-win, :] = features
    x[-win:, :] = features[-1, :]

    delta = np.zeros(x.shape, dtype=np.float32)
    
    filt = np.zeros(2 * win + 1, dtype=np.float32)
    filt[0] = -1
    filt[-1] = 1

    for i in range(features.shape[1]):
        delta[:, i] = np.convolve(features[:, i], filt)

    return delta[win:-win, :]


def _get_sdc_features(cep, d=1, p=3, k=7, **kwargs):
    """
    Compute the Shifted-Delta-Cepstral features for language identification
    
    :param cep: matrix of feature, 1 vector per line
    :param d: represents the time advance and delay for the delta computation
    :param k: number of delta-cepstral blocks whose delta-cepstral 
       coefficients are stacked to form the final feature vector
    :param p: time shift between consecutive blocks.
    
    return: cepstral coefficient concatenated with shifted deltas
    """

    y = np.r_[
        np.resize(cep[0, :], (d, cep.shape[1])),
        cep,
        np.resize(cep[-1, :], (k * 3 + d, cep.shape[1]))
    ]

    delta = _compute_delta(y, win=d)
    sdc = np.empty((cep.shape[0], cep.shape[1] * k))

    idx = np.zeros(delta.shape[0], dtype='bool')
    for ii in range(k):
        idx[d + ii * p] = True
    for ff in range(len(cep)):
        sdc[ff, :] = delta[idx, :].reshape(1, -1)
        idx = np.roll(idx, 1)
    return sdc


def _get_wav_spectrogram(values, sr, winlen=25, winstep=15, n_mels=128, **kwargs):
    # ms to number of samples
    winlen = int((winlen / 1000.0) * sr)
    winstep = int((winstep / 1000.0) * sr)

    S = librosa.feature.melspectrogram(
        y=values,
        sr=sr,
        hop_length=winstep,
        n_fft=winlen,
        n_mels=n_mels,
    )

    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def _get_columns_silence_flags(cur_mfccs, silence_percentile=15, **kwargs):
    percentile = np.percentile(cur_mfccs[0], silence_percentile)
    left_cols = []
    for i, v in enumerate(cur_mfccs[0]):
        if v >= percentile:
            left_cols.append(True)
        else:
            left_cols.append(False)
    return left_cols
        

def get_wav_feature(wav_path, remove_silence=False, **kwargs):
    values, sr = _read_wav(wav_path)
    
#     # N1
#     mfccs = _get_wav_mfcc(values, sr, **kwargs)
#     spectrogram = _get_wav_spectrogram(values, sr, **kwargs)
#     return np.vstack([mfccs, spectrogram])
    
    # N2
    mfccs = _get_wav_mfcc(values, sr, **kwargs)
    sdcs_mfcc = _get_sdc_features(mfccs[:13, :].T, **kwargs).T
    sdcs_d = _get_sdc_features(mfccs[13:26, :].T, **kwargs).T
    sdcs_dd = _get_sdc_features(mfccs[26:39, :].T, **kwargs).T
    res = np.vstack([mfccs, sdcs_mfcc, sdcs_d, sdcs_dd])
    
    if remove_silence:
        flags = _get_columns_silence_flags(mfccs, **kwargs)
        res = res[:, flags]

    return res


def _generator_features_extractor(file_path):
    return get_wav_feature(file_path, d=2, p=1, k=5, mfccd=True, mfccdd=True, remove_silence=True)


class DatasetGenerator(object):
    def __init__(self, class_to_files, batch_size):
        self.batch_size = batch_size
        self.class_to_files = class_to_files
        self.classes = list(set(self.class_to_files))
        
        if batch_size % len(self.class_to_files) != 0:
            raise ValueError(u'batch_size should be devided by number of classes {}'.format(len(class_to_files)))
        self.language_vectors_in_batch = batch_size / len(self.class_to_files)
        
        self.class_to_vector = dict()
        for c, _ in self.class_to_files.iteritems():
            class_vector = [0.0 for _ in xrange(len(self.class_to_files))]
            class_vector[c] = 1.0
            self.class_to_vector[c] = class_vector
        
        self._files_generators_by_class = dict()
        for c, _ in self.class_to_files.iteritems():
            self._files_generators_by_class[c] = self._get_class_generator(c)
        
        self._queue_by_class = defaultdict(list)
    
    def _get_class_generator(self, c):
        random.shuffle(self.class_to_files[c])
        return iter(self.class_to_files[c])

    def _get_next_class_file(self, c):
        gen = self._files_generators_by_class[c]
        try:
            file_path = next(gen)
        except StopIteration as e:
            gen = self._get_class_generator(c)
            file_path = next(gen)
            self._files_generators_by_class[c] = gen
        return file_path
    
    def _get_next_class_vector(self, c):
        if self._queue_by_class[c]:
            return self._queue_by_class[c].pop()
        file_path = self._get_next_class_file(c)
        features = _generator_features_extractor(file_path)
        self._queue_by_class[c].extend(features.T)
        return self._queue_by_class[c].pop()

    def next(self):
        cur_batch_X = []
        cur_batch_Y = []

        for _ in xrange(self.language_vectors_in_batch):
            for c in self.classes:
                vector = None
                while vector is None:
                    try:
                        vector = self._get_next_class_vector(c)
                    except:
                        continue
                        
                cur_batch_X.append(vector)
                cur_batch_Y.append(self.class_to_vector[c])

        return np.array(cur_batch_X), np.array(cur_batch_Y)
    
    def __iter__(self):
        return self


def init_class_to_files_list(folder):
	MAP_LANGUAGE_TO_FILES = defaultdict(list)
	for dirname in os.listdir(folder):
	    all_language_files = os.path.join(folder, dirname, 'all')
	    for filename in os.listdir(all_language_files):
	        full_file_path = os.path.join(all_language_files, filename)
	        MAP_LANGUAGE_TO_FILES[dirname].append(full_file_path)

	LANGUAGE_TO_CLASS = dict(zip(sorted(MAP_LANGUAGE_TO_FILES.keys()), range(len(ONLY_LANGUAGES))))
	MAP_CLASS_TO_FILES_LIST = dict()
	for l, f in MAP_LANGUAGE_TO_FILES.iteritems():
	    MAP_CLASS_TO_FILES_LIST[LANGUAGE_TO_CLASS[l]] = f

	return LANGUAGE_TO_CLASS, MAP_LANGUAGE_TO_FILES, MAP_CLASS_TO_FILES_LIST


print 'Initializing class to files'
LANGUAGE_TO_CLASS, MAP_LANGUAGE_TO_FILES, MAP_CLASS_TO_FILES_LIST = init_class_to_files_list('')
print len(MAP_CLASS_TO_FILES_LIST)


print 'Initializing models'
### instantiate model
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

model.add(Dense(len(MAP_CLASS_TO_FILES_LIST)))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

# model.summary()


def _split_language_files(files_list, train_size, test_size):
    count_by_file_id = defaultdict(int)
    for filename in files_list:
        uid = filename.split('.mp3')[0].split('/')[-1]
        count_by_file_id[uid] += 1

    if len(count_by_file_id) == 3:
        ctrain, ctest, cval = 1, 1, 1
    else:
        count = len(count_by_file_id)
        ctrain = int(count * 0.7)
        ctestval = count - ctrain
        ctest = (ctestval + 1) / 2
        cval = ctestval - ctest

    train_files_uids = [uid for uid, count in sorted(count_by_file_id.iteritems(), key=lambda x: x[1], reverse=True)[:ctrain]]
    test_files_uids = [uid for uid, count in sorted(count_by_file_id.iteritems(), key=lambda x: x[1], reverse=True)[ctrain:ctrain+ctest]]
    val_files_uids = [uid for uid, count in sorted(count_by_file_id.iteritems(), key=lambda x: x[1], reverse=True)[ctrain + ctest:]]

    train_files = []
    test_files = []
    val_files = []

    for filename in files_list:
        uid = filename.split('.mp3')[0].split('/')[-1]
        if uid in train_files_uids:
            train_files.append(filename)
        elif uid in test_files_uids:
            test_files.append(filename)
        elif uid in val_files_uids:
            val_files.append(filename)
    
    return train_files, test_files, val_files


def init_train_val_test_folders_iter(language_to_files_list):
    _TRAIN_CLASS_TO_FILES = defaultdict(list)
    _TEST_CLASS_TO_FILES = defaultdict(list)
    _VAL_CLASS_TO_FILES = defaultdict(list)
    
    for c, files_paths in language_to_files_list.iteritems():
        train_files, test_files, val_files = _split_language_files(files_paths, 0.6, 0.2)
        _TRAIN_CLASS_TO_FILES[c].extend(train_files)
        _TEST_CLASS_TO_FILES[c].extend(test_files)
        _VAL_CLASS_TO_FILES[c].extend(val_files)

    return _TRAIN_CLASS_TO_FILES, _TEST_CLASS_TO_FILES, _VAL_CLASS_TO_FILES


print 'Splitting data'
train_gen_data, test_gen_data, val_gen_data = init_train_val_test_folders_iter(MAP_CLASS_TO_FILES_LIST)


def estimate_total_dataset_duration(class_to_files_list):
    total_files = sum(len(b) for a, b in class_to_files_list.iteritems())
    average_file_duration = 11.0
    vectors_per_file = 500
    return total_files * vectors_per_file


epochs = 4
batch_size = len(train_gen_data) * 4
train_data_vectors = estimate_total_dataset_duration(train_gen_data)
val_data_vectors = estimate_total_dataset_duration(val_gen_data)
train_data_steps = train_data_vectors / batch_size
val_data_steps = val_data_vectors / batch_size
print train_data_vectors, train_data_steps
print val_data_vectors, val_data_steps


history = model.fit_generator(
    DatasetGenerator(train_gen_data, batch_size),
    validation_data=DatasetGenerator(val_gen_data, batch_size),
    validation_steps=val_data_steps,
    steps_per_epoch=train_data_steps,
    epochs=epochs
)
