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
import wave
import contextlib



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


def _get_columns_silence_flags(cur_mfccs, silence_percentile=25, **kwargs):
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
    b1 = get_wav_feature(file_path, d=3, p=1, k=5, mfccd=True, mfccdd=True, remove_silence=True)
    b2 = get_wav_feature(file_path, d=1, p=1, k=7, mfccd=True, mfccdd=True, remove_silence=True)
    return np.vstack([b1, b2])


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
        all_language_files = os.path.join(folder, dirname)
        for filename in os.listdir(all_language_files):
            full_file_path = os.path.join(all_language_files, filename)
            MAP_LANGUAGE_TO_FILES[dirname].append(full_file_path)

    LANGUAGE_TO_CLASS = dict(zip(sorted(MAP_LANGUAGE_TO_FILES.keys()), range(len(MAP_LANGUAGE_TO_FILES))))
    MAP_CLASS_TO_FILES_LIST = dict()
    for l, f in MAP_LANGUAGE_TO_FILES.iteritems():
        MAP_CLASS_TO_FILES_LIST[LANGUAGE_TO_CLASS[l]] = f

    return LANGUAGE_TO_CLASS, MAP_LANGUAGE_TO_FILES, MAP_CLASS_TO_FILES_LIST


print 'Initializing class to files'
LANGUAGE_TO_CLASS, MAP_LANGUAGE_TO_FILES, MAP_CLASS_TO_FILES_LIST = init_class_to_files_list('/home/kolegor/Study/Master/data/use2big_wav_big_splitted_5')
print len(MAP_CLASS_TO_FILES_LIST)


def _get_vectors_in_second():
    # 171 in 3 seconds -> 57 in one second
    return 57

def _get_file_duration(file_path):
    with contextlib.closing(wave.open(file_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def _get_files_list_total_duration(files_list, known_one_file_duration=None):
    if known_one_file_duration is not None:
        return known_one_file_duration * len(files_list)
    return sum(map(_get_file_duration, files_list))

def get_total_dataset_duration(class_to_files, known_one_file_duration=None):
    result = 0.0
    for c, files_list in class_to_files.iteritems():
        print c, len(class_to_files)
        result += _get_files_list_total_duration(files_list, known_one_file_duration=known_one_file_duration)
    return result


print 'Initializing models'
### instantiate model
# model = Sequential()

# # we can think of this chunk as the input layer
# model.add(Dense(200, input_dim=546, init='uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# # we can think of this chunk as the hidden layer    
# model.add(Dense(128, init='uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.25))

# # we can think of this chunk as the hidden layer    
# model.add(Dense(64, init='uniform'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Dense(128, use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.2))

# model.add(Dense(80, use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Dense(len(MAP_CLASS_TO_FILES_LIST)))
# model.add(Activation('softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.RMSprop(lr=0.001),
#               metrics=['accuracy'])

model = load_model('/home/kolegor/model.big_11_lang__bottleneck_nn__features_sdc_315_and_sdc_117.keras')

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


def estimate_total_dataset_duration(class_to_files_list, known_one_file_duration=None):
    total_duration = get_total_dataset_duration(class_to_files_list, known_one_file_duration=known_one_file_duration)
    vectors_in_second = _get_vectors_in_second()
    return total_duration * vectors_in_second


if False:
    epochs = 2
    batch_size = len(train_gen_data) * 3
    train_data_vectors = estimate_total_dataset_duration(train_gen_data, known_one_file_duration=5)
    val_data_vectors = estimate_total_dataset_duration(val_gen_data, known_one_file_duration=5)
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

    model.save('/home/kolegor/model.big_11_lang__bottleneck_nn__features_sdc_315_and_sdc_117.keras')
else:

    def _get_file_uid(file_path):
        return file_path.split('.mp3')[0].split('/')[-1]

    # tuples (file_path, real_class, predicted_class)
    all_predictions = []
    all_files_count = sum(len(b) for a, b in test_gen_data.iteritems())

    i = 0
    count_correct = 0
    count_all = 0

    for cl, files_paths in test_gen_data.iteritems():
        for file_path in files_paths:
            if i % 100 == 0:
                print i, all_files_count
            i += 1

            try:
                features = _generator_features_extractor(file_path)
                cur_predictions = model.predict(features.T)

                for column, p in zip(features.T, cur_predictions):
                    pcl = np.argmax(p)
                    count_all += 1
                    if pcl == cl:
                        count_correct += 1
                    all_predictions.append((file_path, cl, list(p), pcl))
            except KeyboardInterrupt:
                break
            except:
                continue

    print count_correct, count_all, float(count_correct) / count_all

    try:
        import json
        with open('/home/kolegor/predictions2.json', 'w') as outf:
            json.dump(all_predictions, outf)
    except:
        # print all_predictions
        pass

    print count_correct, count_all, float(count_correct) / count_all

    # predictions_by_file_id = defaultdict(list)
    # real_by_file_id = dict()
    # classes = set()
    # for x, file_id, pred, real in zip(all_test_X, all_test_X_file_uid, predictions, all_test_Y):
    #     predictions_by_file_id[file_id].append(pred)
    #     real_class = np.argmax(real)
    #     if file_id in real_by_file_id and real_by_file_id[file_id] != real_class:
    #         raise Exception('!!!')
    #     real_by_file_id[file_id] = real_class
    #     classes.add(MAP_CLASS_TO_LANGUAGE[real_class])

    # confusion_matrix = np.zeros((len(classes), len(classes)))
    # overall_count, overall_true = 0, 0
    # for file_id, file_predictions in predictions_by_file_id.iteritems():
    #     real_class = real_by_file_id[file_id]
        
    #     pred_class = None
        
    #     # 1 - Max sure over all vectors
    #     if False:
    #         _max_sure = -1
    #         _max_sure_class = None
    #         for p in file_predictions:
    #             if max(p) > _max_sure:
    #                 _max_sure = max(p)
    #                 _max_sure_class = np.argmax(p)
    #         pred_class = _max_sure_class
        
    #     # 2 - Max count over all vectors
    #     else:
    #         counter = Counter()
    #         for p in file_predictions:
    #             counter[np.argmax(p)] += 1
    #         pred_class = counter.most_common(1)[0][0]
        
    #     overall_count += 1
    #     if real_class == pred_class:
    #         overall_true += 1

    #     confusion_matrix[real_class][pred_class] += 1

    # print float(overall_true) / overall_count
    # confusion_matrix = confusion_matrix.astype(int)
    # print confusion_matrix