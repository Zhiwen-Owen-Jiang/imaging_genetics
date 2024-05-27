import os
import logging
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from heig.sumstats import read_sumstats
from heig.voxelgwas import (
    check_input,
    recover_se
)


MAIN_DIR = os.path.join(os.getcwd(), 'test', 'test_secondary')
log = logging.getLogger()
log.setLevel(logging.INFO)


class Args:
    def __init__(self, ldr_sumstats=None, bases=None, inner_ldr=None,
                 extract=None, n_ldrs=None, voxel=None, range=None,
                 sig_thresh=None):
        self.ldr_sumstats = ldr_sumstats
        self.bases = bases
        self.inner_ldr = inner_ldr
        self.extract = extract
        self.n_ldrs = n_ldrs
        self.voxel = voxel
        self.range = range
        self.sig_thresh = sig_thresh


class Test_check_input(unittest.TestCase):
    def test_good_cases(self):
        # voxel
        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    voxel=1)
        start_chr, start_pos, end_pos, _ = check_input(args, log)
        self.assertEqual(start_chr, None)
        self.assertEqual(start_pos, None)
        self.assertEqual(end_pos, None)

        # range
        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    range="3:1,3:3")
        start_chr, start_pos, end_pos, _ = check_input(args, log)
        self.assertEqual(start_chr, 3)
        self.assertEqual(start_pos, 1)
        self.assertEqual(end_pos, 3)

        # extract
        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    extract=[os.path.join(MAIN_DIR, 'extract.txt')])
        *_, keep_snps = check_input(args, log)
        true_keep_snps = pd.DataFrame({'SNP': ['rs2']})
        assert_frame_equal(keep_snps, true_keep_snps)

    def test_bad_cases(self):
        # missing required arguments
        def exclude_key(d, key_to_exclude):
            return {key: value for key, value in d.items() if key != key_to_exclude}
        args_dict = {'ldr_sumstats': os.path.join(MAIN_DIR, 'gwas'),
                     'bases': os.path.join(MAIN_DIR, 'bases.npy'),
                     'inner_ldr': os.path.join(MAIN_DIR, 'inner_ldr.npy')}
        for key in args_dict.keys():
            args_dict = exclude_key(args_dict, key)
            args = Args(**args_dict)
            with self.assertRaises(ValueError):
                check_input(args, log)

        # wrong specifications of range
        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    range="3:1,4:3")
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    range="3:5,3:3")
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    range="3:5;3:3")
        with self.assertRaises(ValueError):
            check_input(args, log)

        # invalid arguments
        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    voxel=0)
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    sig_thresh=0)
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
                    bases=os.path.join(MAIN_DIR, 'bases.npy'),
                    inner_ldr=os.path.join(MAIN_DIR, 'inner_ldr.npy'),
                    n_ldrs=0)
        with self.assertRaises(ValueError):
            check_input(args, log)


class Test_recover_se(unittest.TestCase):
    def test_recover_se(self):
        bases = np.load(os.path.join(MAIN_DIR, 'bases.npy'))[:, :3]
        inner_ldr = np.load(os.path.join(MAIN_DIR, 'inner_ldr.npy'))[:3, :3]
        sumstats = read_sumstats(os.path.join(MAIN_DIR, 'gwas'))
        n = np.array(sumstats.snpinfo['N']).reshape(-1, 1)[:4, ]
        ldr_beta = sumstats.beta[:4, ]
        ztz_inv = np.ones(4).reshape(4, 1)

        se = recover_se(bases[0], inner_ldr, n, ldr_beta, ztz_inv).reshape(-1)
        voxel_beta = np.dot(ldr_beta, bases[0].T)
        self.assertEqual(se.shape, voxel_beta.shape)

        se = recover_se(bases, inner_ldr, n, ldr_beta, ztz_inv)
        voxel_beta = np.dot(ldr_beta, bases.T)
        self.assertEqual(se.shape, voxel_beta.shape)
