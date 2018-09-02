# coding=utf-8

import os


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '../../data/')


LANGUAGES_WAV_DIR = os.path.join(DATA_DIR, 'clean/')
WAV_LANGUAGE_PATH_TEMPLATE = LANGUAGES_WAV_DIR + u'/{language}/'
WAV_LANGUAGE_DATASET_PATH_TEMPLATE = LANGUAGES_WAV_DIR + u'/{language}/{dataset}/'
WAV_PATH_TEMPLATE = LANGUAGES_WAV_DIR + u'/{language}/{dataset}/{filename}.wav'

FEATURES_DIR = os.path.join(DATA_DIR, 'features/')
FEATURE_LANGUAGE_DIR_TEMPLATE = FEATURES_DIR + u'/{feature}/{language}/{dataset}/'
FEATURE_LANGUAGES_PATH_TEMPLATE = FEATURES_DIR + u'/{feature}/{language}/{dataset}/{filename}{extension}'
