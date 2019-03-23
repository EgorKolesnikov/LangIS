# coding=utf-8

import os
from multiprocessing import Process
import traceback

from src import constants
from src.languages import Register
import handlers


def get_folder_files(folder):
    all_entries = os.listdir(folder)
    result = []

    for filename in all_entries:
        if not filename.endswith('.wav'):
            continue
        result.append(os.path.join(folder, filename))

    return result


def get_file_credentials(full_wav_path):
    """ Should be language/dataset/filename.wav"""
    tokens = full_wav_path.split('/')
    return tokens[-3], tokens[-2], u'.'.join(tokens[-1].split('.')[:-1])


def get_all_files(languages, datasets):
    all_files = []

    for language in languages:
        for dataset in datasets:
            folder = constants.WAV_LANGUAGE_DATASET_PATH_TEMPLATE.format(
                language=language,
                dataset=dataset
            )

            all_files.extend(get_folder_files(folder))

    return all_files


def create_and_save_feature(wav_files, feature_handler, uid=None):
    print u'START PROC # {}. COUNT: {}'.format(uid, len(wav_files))

    for i, wav_path in enumerate(wav_files):
        if i % 100 == 0:
            print u'PROC # {}. {}/{}'.format(uid, i, len(wav_files))

        language, dataset, filename = get_file_credentials(wav_path)
        target_path = feature_handler.target_path(language, dataset, filename)

        if os.path.exists(target_path):
            continue

        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path))

        try:
            feature_handler.wav_to_feature(wav_path, target_path)
        except:
            # print traceback.format_exc()
            print u' - PROC # {}. EXCEPTION {}'.format(uid, wav_path)

        # print language, dataset, filename, target_path


def bulk_create(languages, datasets, feature_handler, processes=4):
    all_files = get_all_files(languages, datasets)
    print u'Found {} files in {} languages. Datasets: {}'.format(len(all_files), len(languages), datasets)

    procs = []

    chunk_size = (len(all_files) + processes - 1) / processes
    for chunk in xrange(processes):
        chunk_files = all_files[chunk * chunk_size:min(len(all_files), (chunk + 1) * chunk_size)]

        proc = Process(target=create_and_save_feature, args=(chunk_files, feature_handler, chunk))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()


_use_languages = [
    'ab_Danish',
    'ab_Hebrew',
    'ab_Finnish',
    'ab_Poland',
    # 'khmer',
    # 'nepali',
    # 'sa_afrikaans',
    # 'sa_isiXhosa',
    # 'sa_sesotho',
    # 'sa_setswana',
]

bulk_create(_use_languages, ['train', 'test', 'dev'], handlers.MelSpectrogramFeature)
# bulk_create(_use_languages, ['train', 'test', 'dev'], handlers.MFCCFeature)
