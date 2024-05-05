import os
import unittest

from heig import utils


MAIN_DIR = os.getcwd()


class Test_check_compression(unittest.TestCase):
    def setUp(self):
        self.good_file_compression = {
            'test.bgz': 'gzip',
            'test.gz': 'gzip',
            'test.bz2': 'bz2'
        }

        self.bad_files = [
            'test.tar',
            'test.tar.gz',
            'test.tar.bz2',
            'test.zip'
        ]

        self.folder = os.path.join(
            MAIN_DIR, 'test', 'test_utils', 'compressed_files')

    def test_compression(self):
        for file, comp in self.good_file_compression.items():
            # print(file)
            file_dir = os.path.join(self.folder, file)
            openfunc, comp_ = utils.check_compression(file_dir)
            self.assertEqual(comp_, comp)

            with openfunc(file_dir, 'r') as file_:
                header = file_.readline().split()
            # print(header)
            self.assertEqual(str(header[0], 'UTF-8'), 'FID')
            self.assertEqual(str(header[1], 'UTF-8'), 'IID')
            self.assertEqual(str(header[2], 'UTF-8'), 'x')
            self.assertEqual(str(header[3], 'UTF-8'), '1')

        with self.assertRaises(ValueError):
            for file in self.bad_files:
                file_dir = os.path.join(self.folder, file)
                utils.check_compression(file_dir)
