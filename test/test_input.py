import os
import unittest
import pandas as pd

from heig.input.dataset import read_keep, read_geno_part, read_extract


MAIN_DIR = os.getcwd()


class Test_read_keep(unittest.TestCase):
    def setUp(self):
        folder = os.path.join(MAIN_DIR, 'test', 'test_input', 'keep_files')
        
        self.keep_files = ['keep1.txt',
                           'keep2.txt',
                           'keep3.txt',
                           'keep_empty.txt']
        self.keep_files = [os.path.join(folder, x) for x in self.keep_files]
        
        self.bad_keep_files = ['keep_bad1.txt']
        self.bad_keep_files = [os.path.join(folder, x) for x in self.bad_keep_files]    

    def test_read_keep(self):
        true_value = pd.MultiIndex.from_arrays([('s2', 's3'), ('s2', 's3')], 
                                               names=['FID', 'IID'])
        self.assertTrue(read_keep(self.keep_files).equals(true_value))
        
        with self.assertRaises(ValueError):
            read_keep(self.bad_keep_files)



class Test_read_extract(unittest.TestCase):
    def setUp(self):
        folder = os.path.join(MAIN_DIR, 'test', 'test_input', 'extract_files')
        self.good_files = ['extract1.txt',
                           'extract2.txt',
                           'extract3.txt']
        self.good_files = [os.path.join(folder, x) for x in self.good_files]

        self.bad_files = ['extract1.txt', 'extract4.txt']
        self.bad_files = [os.path.join(folder, x) for x in self.bad_files]

    def test_read_extract(self):
        true_value = pd.DataFrame({'SNP': ['rs2']})
        self.assertTrue(read_extract(self.good_files).equals(true_value))

        with self.assertRaises(ValueError):
            read_extract(self.bad_files)

