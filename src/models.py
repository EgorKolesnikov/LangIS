# coding=utf-8

from collections import defaultdict, Counter
from keras.models import load_model
import numpy as np
import random

import util


class AbstractDatasetGenerator(object):
    """
    Using dataset on train stage directly from disk.
    (loading by batches in memory)
    """
    # Number of features vectors
    # in one second of input file
    VECTORS_IN_SECOND = 60

    def __init__(self, class_to_files, batch_size, one_file_duration=None):
        self.batch_size = batch_size
        self.class_to_files = class_to_files
        self.one_file_duration = one_file_duration

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
        features = self.get_file_features(file_path)
        self._queue_by_class[c].extend(features.T)

        return self._queue_by_class[c].pop()

    def get_file_features(self, file_path, **kwargs):
        raise NotImplementedError

    def next(self):
        cur_batch_x = []
        cur_batch_y = []

        for _ in xrange(self.language_vectors_in_batch):
            for c in self.classes:
                vector = None
                while vector is None:
                    try:
                        vector = self._get_next_class_vector(c)
                    except:
                        continue

                cur_batch_x.append(vector)
                cur_batch_y.append(self.class_to_vector[c])

        return np.array(cur_batch_x), np.array(cur_batch_y)

    def estimate_size(self):
        """
        To be able to use dataset generator with keras, we need to tell it how much
        batches it should expect on train and val epoch.
        Because we can not load all data in memory, we need to estimate number of files/vectors/bathes
        we are going to load.
        :return: total dataset duration in seconds
        """
        total_duration = util.sum_dataset_duration(self.class_to_files, known_one_file_duration=self.one_file_duration)
        return total_duration * self.VECTORS_IN_SECOND

    def estimate_batches(self):
        return self.estimate_size() / self.batch_size

    def get_all_files(self):
        result = []
        for _, files in self.class_to_files.iteritems():
            result.extend(files)
        return result

    def __iter__(self):
        return self


def estimate_total_dataset_duration(class_to_files_list, known_one_file_duration=None, vectors_in_second=60):
    """
    To be able to use dataset generator with keras, we need to tell it how much
    batches it should expect on train and val epoch.
    Because we can not load all data in memory, we need to estimate number of files/vectors/bathes
    we are going to load.
    :param class_to_files_list: dict(class uid -> list of language files absolute paths)
    :param known_one_file_duration: duration in seconds of one file
    :param vectors_in_second: number of features vectors in one second of input file
    :return: number of total vectors over all dataset
    """
    total_duration = util.sum_dataset_duration(class_to_files_list, known_one_file_duration=known_one_file_duration)
    return total_duration * vectors_in_second


def train(model, train_generator, val_generator, save_to, epochs=3):
    """
    Train specified model with specified train and validation datasets.
    :param model: keras model object
    :param train_generator: AbstractDatasetGenerator object (train data)
    :param val_generator: AbstractDatasetGenerator object (validation data)
    :param save_to: path to save trained model
    :param epochs: number of epochs to run
    :return: keras.train() result (history object)
    """
    # LANGUAGE_TO_CLASS, MAP_LANGUAGE_TO_FILES, MAP_CLASS_TO_FILES_LIST = init_class_to_files_list(
    #     '/home/kolegor/Study/Master/data/use2big_wav_big_splitted_5')
    #
    # train_gen_data, test_gen_data, val_gen_data = init_train_val_test_folders_iter(MAP_CLASS_TO_FILES_LIST)

    train_data_steps = train_generator.estimate_batches()
    val_data_steps = val_generator.estimate_batches()

    history = model.fit_generator(
        train_generator,  # DatasetGenerator(train_gen_data, batch_size),
        validation_data=val_generator,  # DatasetGenerator(val_gen_data, batch_size),
        validation_steps=val_data_steps,
        steps_per_epoch=train_data_steps,
        epochs=epochs
    )

    model.save(save_to)
    return history


def test(model_path, test_generator):
    """
    Run test on specified model using specified dataset.
    :param model_path: path to saved keras model
    :param test_generator: AbstractDatasetGenerator object (validation data)
    :return: tuple(accuracy by vectors, accuracy by files)
    """
    model = load_model(model_path)

    all_predictions = []
    all_predictions_by_file = defaultdict(list)

    count_correct = 0
    count_all = 0

    for cl, files_paths in test_generator.class_to_files.iteritems():
        for file_path in files_paths:
            try:
                features = test_generator.get_file_features(file_path)
                cur_predictions = model.predict(features.T)

                for column, p in zip(features.T, cur_predictions):
                    pcl = np.argmax(p)
                    count_all += 1
                    if pcl == cl:
                        count_correct += 1

                    all_predictions.append((file_path, cl, tuple(list(p)), pcl))
                    all_predictions_by_file[file_path].append((cl, pcl))
            except:
                continue

    acc_vectors = float(count_correct) / count_all
    print count_correct, count_all, acc_vectors

    count_correct_files = 0
    for file_path, file_predictions in all_predictions_by_file.iteritems():
        real_class = file_predictions[0][0]
        most_common = Counter([pcl for _, pcl in file_predictions]).most_common(1)[0][0]
        if most_common == real_class:
            count_correct_files += 1

    acc_files = 100.0 * count_correct_files / len(all_predictions_by_file)
    print count_correct_files, len(all_predictions_by_file), acc_files

    return count_correct, count_all, count_correct_files
