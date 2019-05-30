# coding=utf-8

from collections import defaultdict
import contextlib
import os
import random
import wave


def load_dataset_meta(folder):
    """
    Parsing dataset folder. Group input files by language name.
    Assign each language UID.
    :param folder: base dataset folder absolute path.
    :return: tuple of (
        dict(language lower name -> class UID (starting from 0))
        dict(language lower name -> list of absolute paths of all language files)
        dict(language class UID -> list of absolute paths of all language files)
    )
    """
    language_to_files = defaultdict(list)

    for dir_name in os.listdir(folder):
        all_language_files = os.path.join(folder, dir_name)
        for filename in os.listdir(all_language_files):
            full_file_path = os.path.join(all_language_files, filename)
            language_to_files[dir_name].append(full_file_path)

    language_to_class = dict(zip(
        sorted(language_to_files.keys()),
        range(len(language_to_files)))
    )

    class_to_files = dict()
    for l, f in language_to_files.iteritems():
        class_to_files[language_to_class[l]] = f

    return language_to_class, language_to_files, class_to_files


def get_file_duration(file_path):
    with contextlib.closing(wave.open(file_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def sum_files_duration(files_list, known_one_file_duration=None):
    if known_one_file_duration is not None:
        return known_one_file_duration * len(files_list)
    return sum(map(get_file_duration, files_list))


def sum_dataset_duration(class_to_files, known_one_file_duration=None):
    result = 0.0
    for c, files_list in class_to_files.iteritems():
        result += sum_files_duration(files_list, known_one_file_duration=known_one_file_duration)
    return result


def split_language_files(files_list, train_size, test_size):
    """
    Split dataset language files into train/test/val.
    :param files_list: list of all files
    :param train_size: size of train part [0.0 - 1.0]
    :param test_size: size if test part [0.0 - 1.0]
    :return: 3 lists: train/test/val files
    """
    count_by_file_id = defaultdict(int)

    random.seed(42)
    random.shuffle(files_list)

    for filename in files_list:
        uid = filename.split('.mp3')[0].split('/')[-1]
        count_by_file_id[uid] += 1

    if len(count_by_file_id) == 3:
        ctrain, ctest = 1, 1
    else:
        count = len(count_by_file_id)
        ctrain = int(count * train_size)
        ctest = int(count * test_size)

    def sort_key(x):
        return x[1]

    train_uids = [uid for uid, count in sorted(count_by_file_id.iteritems(), key=sort_key, reverse=True)[:ctrain]]
    test_uids = [uid for uid, count in sorted(count_by_file_id.iteritems(), key=sort_key, reverse=True)[ctrain:ctrain + ctest]]
    val_uids = [uid for uid, count in sorted(count_by_file_id.iteritems(), key=sort_key, reverse=True)[ctrain + ctest:]]

    train_files = []
    test_files = []
    val_files = []

    for filename in files_list:
        uid = filename.split('.mp3')[0].split('/')[-1]
        if uid in train_uids:
            train_files.append(filename)
        elif uid in test_uids:
            test_files.append(filename)
        elif uid in val_uids:
            val_files.append(filename)

    return train_files, test_files, val_files


def init_train_val_test_folders_iter(language_to_files_list):
    train_class_to_files = defaultdict(list)
    test_class_to_files = defaultdict(list)
    val_class_to_files = defaultdict(list)

    for c, files_paths in language_to_files_list.iteritems():
        train_files, test_files, val_files = split_language_files(files_paths, 0.6, 0.2)
        train_class_to_files[c].extend(train_files)
        test_class_to_files[c].extend(test_files)
        val_class_to_files[c].extend(val_files)

    return train_class_to_files, test_class_to_files, val_class_to_files
