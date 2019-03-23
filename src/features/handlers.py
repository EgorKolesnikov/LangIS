# coding=utf-8

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.image as mpimg

from src import constants


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class BaseFeatureHandler(object):
    UID = None
    EXTENSION = None

    @classmethod
    def _read_wav(cls, file_path):
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

    @classmethod
    def wav_to_feature(cls, wav_path, save_path, **kwargs):
        raise NotImplementedError('Needs to be overridden')

    @classmethod
    def load_feature(cls, feature_path, **kwargs):
        raise NotImplementedError('Needs to be overridden')

    @classmethod
    def target_path(cls, language, dataset, filename):
        return constants.FEATURE_LANGUAGES_PATH_TEMPLATE.format(
            feature=cls.UID,
            language=language,
            dataset=dataset,
            filename=filename,
            extension=cls.EXTENSION
        )


class MelSpectrogramFeature(BaseFeatureHandler):
    UID = 'spectrogram'
    EXTENSION = '.png'

    @classmethod
    def wav_to_feature(cls, wav_path, save_path, winlen=25, winstep=15, n_mels=128, **kwargs):
        y, sr = cls._read_wav(wav_path)

        # ms to number of samples
        winlen = int((winlen / 1000.0) * sr)
        winstep = int((winstep / 1000.0) * sr)

        S = librosa.feature.melspectrogram(
            y=y, sr=sr,
            hop_length=winstep,
            n_fft=winlen,
            n_mels=n_mels,
        )

        log_S = librosa.power_to_db(S, ref=np.max)

        sizes = np.shape(log_S)
        height = float(sizes[0])
        width = float(sizes[1])

        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        save_path = u'{path}#{sr}={winlen}={winstep}{ext}'.format(
            path=save_path,
            sr=sr,
            winlen=winlen,
            winstep=winstep,
            ext=cls.EXTENSION
        )

        ax.imshow(log_S, cmap='gray', origin='lower')
        plt.savefig(save_path, dpi=height)
        plt.close()

    @classmethod
    def load_feature(cls, feature_path, **kwargs):
        img = mpimg.imread(feature_path)
        return rgb2gray(img)


class MFCCFeature(BaseFeatureHandler):
    UID = 'mfcc'
    EXTENSION = '.txt'

    @classmethod
    def wav_to_feature(
            cls,
            wav_path,
            save_path,
            winlen=25,
            winstep=15,
            n_mels=128,
            n_mfcc=13,
            need_d=True,
            need_dd=True,
            fmin=0,
            fmax=6000,
            **kwargs
    ):
        y, sr = cls._read_wav(wav_path)

        # ms to number of samples
        winlen = int((winlen / 1000.0) * sr)
        winstep = int((winstep / 1000.0) * sr)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_fft=winlen,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            hop_length=winstep,
            fmin=0,
            fmax=8000
        )

        blocks = [mfcc]

        if need_d:
            d = librosa.feature.delta(mfcc)
            blocks.append(d)
        if need_dd:
            dd = librosa.feature.delta(mfcc, order=2)
            blocks.append(dd)

        features = np.vstack(blocks)

        save_path = u'{path}#{sr}={winlen}={winstep}={n_mfcc}={n_mels}={d}={dd}'.format(
            path=save_path,
            sr=sr,
            winlen=winlen,
            winstep=winstep,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            d=int(need_d),
            dd=int(need_dd)
        )

        np.savetxt(save_path, features, fmt='%.9e')

    @classmethod
    def load_feature(cls, feature_path, **kwargs):
        return np.loadtxt(feature_path)
