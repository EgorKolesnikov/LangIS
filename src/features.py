# coding=utf-8

import librosa
import librosa.display
import numpy as np
import wavfile


def read_wav(file_path):
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
        y, sr = librosa.load(file_path, sr=None)
        return y, sr


def get_columns_silence_flags(cur_mfccs, silence_percentile=25, **kwargs):
    percentile = np.percentile(cur_mfccs[0], silence_percentile)
    left_cols = []
    for i, v in enumerate(cur_mfccs[0]):
        if v >= percentile:
            left_cols.append(True)
        else:
            left_cols.append(False)
    return left_cols


def delta(features, win=3, **kwargs):
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


def mfcc(values, sr, winlen=25, winstep=15, n_mels=128, n_mfcc=13, mfccd=False, mfccdd=False, norm_mfcc=False, fmin=0, fmax=6000, **kwargs):
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


def sdc(cep, d=1, p=3, k=7, **kwargs):
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

    delta_features = delta(y, win=d)
    sdc_features = np.empty((cep.shape[0], cep.shape[1] * k))

    idx = np.zeros(delta_features.shape[0], dtype='bool')
    for ii in range(k):
        idx[d + ii * p] = True
    for ff in range(len(cep)):
        sdc_features[ff, :] = delta_features[idx, :].reshape(1, -1)
        idx = np.roll(idx, 1)
    return sdc_features


def spectrogram(values, sr, winlen=25, winstep=15, n_mels=128, **kwargs):
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


