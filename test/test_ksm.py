import os
import unittest
import logging
import pandas as pd
from pandas.testing import assert_index_equal

from heig.ksm import (
    get_image_list,
)

MAIN_DIR = os.getcwd()
log = logging.getLogger()
log.setLevel(logging.INFO)


class Test_get_image_list(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.folder = os.path.join(MAIN_DIR, 'test', 'test_ksm')

    def test_single_dir(self):
        true_ids = pd.MultiIndex.from_arrays([('s1000', 's1002', 's1004'), ('s1000', 's1002', 's1004')],
                                             names=['FID', 'IID'])
        true_img_list = ['s1000_example_image.nii.gz',
                         's1002_example_image.nii.gz',
                         's1004_example_image.nii.gz']
        true_img_list = [os.path.join(
            self.folder, 'image_dir1', x) for x in true_img_list]

        ids, img_list = get_image_list([os.path.join(self.folder, 'image_dir1')],
                                       ['_example_image.nii.gz'],
                                       log)
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

        # test duplicated subjects
        ids, img_list = get_image_list([os.path.join(self.folder, 'image_dir1'), 
                                        os.path.join(self.folder, 'image_dir3')],
                                       ['_example_image.nii.gz', '_example_image.nii.gz'],
                                       log)
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

    def test_multiple_dirs(self):
        true_ids = pd.MultiIndex.from_arrays([('s1000', 's1001', 's1002', 's1003', 's1004', 's1005'), 
                                              ('s1000', 's1001', 's1002', 's1003', 's1004', 's1005')],
                                             names=['FID', 'IID'])
        true_img_list = ['image_dir1/s1000_example_image.nii.gz',
                         'image_dir2/s1001_example_image.nii.gz',
                         'image_dir1/s1002_example_image.nii.gz',
                         'image_dir2/s1003_example_image.nii.gz',
                         'image_dir1/s1004_example_image.nii.gz',
                         'image_dir2/s1005_example_image.nii.gz']
        true_img_list = [os.path.join(self.folder, x) for x in true_img_list]

        ids, img_list = get_image_list([os.path.join(self.folder, 'image_dir1'), 
                                        os.path.join(self.folder, 'image_dir2')],
                                       ['_example_image.nii.gz', '_example_image.nii.gz'],
                                       log)
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

    def test_keep(self):
        true_ids = pd.MultiIndex.from_arrays([('s1000', 's1001'), 
                                              ('s1000', 's1001')],
                                             names=['FID', 'IID'])
        true_img_list = ['image_dir1/s1000_example_image.nii.gz',
                         'image_dir2/s1001_example_image.nii.gz',
                        ]
        true_img_list = [os.path.join(self.folder, x) for x in true_img_list]

        keep_indivs = pd.MultiIndex.from_arrays([('s1000', 's1001'), 
                                              ('s1000', 's1001')],
                                             names=['FID', 'IID'])
        ids, img_list = get_image_list([os.path.join(self.folder, 'image_dir1'), 
                                        os.path.join(self.folder, 'image_dir2')],
                                       ['_example_image.nii.gz', '_example_image.nii.gz'],
                                       log, keep_indivs)
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

    def test_one_image(self):
        true_ids = pd.MultiIndex.from_arrays([('s1000',), 
                                              ('s1000',)],
                                             names=['FID', 'IID'])
        true_img_list = ['image_dir3/s1000_example_image.nii.gz',
                        ]
        true_img_list = [os.path.join(self.folder, x) for x in true_img_list]

        ids, img_list = get_image_list([os.path.join(self.folder, 'image_dir3'), ],
                                       ['_example_image.nii.gz', ],
                                       log)
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

    
class Test_load_nifti(unittest.TestCase):
    def test_load_nifti(self):
        folder = os.path.join(MAIN_DIR, 'test', 'test_ksm')
