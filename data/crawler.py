# coding=utf-8

from bs4 import BeautifulSoup
import os
import random
import requests
import shutil
import tarfile
import time
import urllib2


# where to save different languages wav files
LANGUAGES_WAV_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'wav')


# list of tuples(foxforge folder name, local folder name, ru alias)
VOXFORGE_DIRECTORIES = [
    ('Russian', 'ru', u'Русский'),
    ('bg', 'bg', u'Болгарский'),
    ('ca', 'ca', u'Канадский'),
    ('de', 'de', u'Немецкий'),
    ('es', 'es', u'Испанский'),
    ('fr', 'fr', u'Французский'),
    ('hr', 'hr', u'Хорватский'),
    ('it', 'it', u'Итальянский'),
    ('pt', 'pt', u'Португальский'),
    ('tr', 'tr', u'Турецкий'),
    ('zh', 'zh', u'Русский'),
    ('uk', 'uk', u'Украинский'),

    ('el', 'el', u'? Греческий'),
    ('fa', 'fa', u'? Персидский'),
    ('he', 'he', u'? Иврит'),
    ('sq', 'sq', u'? Албанский'),
]


class VoxforgeCrawler(object):
    """
    Load wav files with different languages in local folder
    Using only 48kHz_16bit format of each language
    """
    def __init__(self):
        self.base_url = 'http://www.repository.voxforge1.org/downloads/'
        self.in_dir_path = '/Trunk/Audio/Original/48kHz_16bit/'

        self.limit_language_total_files = 200

    def get_files_names(self, page_url):
        """ Get list of files names in one language folder """
        soup = BeautifulSoup(urllib2.urlopen(page_url).read(), 'html.parser')
        all_links = soup.findAll('a')

        download_names = []

        for link in all_links:
            href = link['href']
            text = link.text

            if href != text:
                continue

            download_names.append(text)

        return download_names

    def save_files(self, page_url, download_names, save_dir, random_load=True):
        """ Download specified wav files from one voxforge language folder """
        count_loaded_files = 0
        count_loaded_size = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if random_load:
            random.shuffle(download_names)

        # where to save and unpack downloaded files
        temp_package_path = os.path.join(save_dir, 'temp.tgz')
        temp_package_upacked_dir = os.path.join(save_dir, 'temp_dir')

        one_package_size = 10.0
        approx_files_count = len(download_names) * one_package_size
        limit_one_package_files = int(max(3.0, self.limit_language_total_files / approx_files_count * one_package_size))

        for idx_package, package_to_load in enumerate(download_names):
            print 'Current language files: {}. Package to load: {} (limit package: {})'.format(
                count_loaded_files, package_to_load, limit_one_package_files
            )

            # leave if we have enough files
            is_have_enough = count_loaded_files >= self.limit_language_total_files
            if is_have_enough:
                break

            package_name_to_load = package_to_load.split('.')[0]

            # download to temp file
            url = '{}{}'.format(page_url, package_to_load)
            r = requests.get(url, allow_redirects=True)
            open(temp_package_path, 'wb').write(r.content)

            time.sleep(1)

            # unzip tgz in temp dir
            tar = tarfile.open(temp_package_path, "r:gz")
            tar.extractall(path=temp_package_upacked_dir)
            full_unzipped_path = os.path.join(temp_package_upacked_dir, package_name_to_load, 'wav')

            count_loaded_size += sum(info.size for info in tar.getmembers())

            # if not 'wav' dir in package - skip package
            if not os.path.exists(full_unzipped_path):
                continue

            # move wav files from unzipped folder to local languages storage
            for idx_file_in_package, filename_to_move in enumerate(os.listdir(full_unzipped_path)):
                # full path of file to move
                what_to_move = os.path.join(full_unzipped_path, filename_to_move)

                # can not leave previous name (different packages may have files with equal names)
                where_to_move = os.path.join(save_dir, '{}__{}'.format(package_name_to_load, filename_to_move))

                # check limits
                is_fit_package_limit = idx_file_in_package + 1 <= limit_one_package_files
                is_fit_language_limit = count_loaded_files <= self.limit_language_total_files

                if not is_fit_package_limit or not is_fit_language_limit:
                    break

                # move file
                os.rename(what_to_move, where_to_move)
                count_loaded_files += 1

            # cleanup
            shutil.rmtree(temp_package_upacked_dir)
            tar.close()

        os.remove(temp_package_path)
        return count_loaded_files, count_loaded_size

    def run(self):
        total_size = 0
        total_count = 0

        for idx_language, (dir_name, save_dir_name, language_name) in enumerate(VOXFORGE_DIRECTORIES):
            url = self.base_url + dir_name + self.in_dir_path
            print u'\n{}/{}. {:<20} {}'.format(idx_language + 1, len(VOXFORGE_DIRECTORIES), language_name, url)

            files_names = self.get_files_names(url)
            print 'Found: {} packages'.format(len(files_names))

            dir_to_save = os.path.join(LANGUAGES_WAV_DIR, save_dir_name)
            loaded_count, loaded_size = self.save_files(url, files_names, dir_to_save)
            print 'Packages loaded: {}. Total size: {} (MB)'.format(loaded_count, loaded_size / 1e6)

            total_count += loaded_count
            total_size += loaded_size

        print 'Downloaded total: {} packages {} languages ({} MB)'.format(
            total_count,
            len(VOXFORGE_DIRECTORIES),
            total_size / 1e6
        )


if __name__ == "__main__":
    crawler = VoxforgeCrawler()
    crawler.run()
