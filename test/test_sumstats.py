import os
import logging
import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from heig.sumstats import (
    check_input,
    map_cols,
    read_sumstats,
    GWAS,
)

MAIN_DIR = os.path.join(os.getcwd(), 'test', 'test_sumstats')
log = logging.getLogger()
log.setLevel(logging.INFO)


class Args:
    def __init__(self, snp_col=None, a1_col=None, a2_col=None, n_col=None, effect_col=None,
                 ldr_gwas=None, se_col=None, chr_col=None, pos_col=None,
                 z_col=None, p_col=None, maf_col=None, maf_min=None,
                 info_col=None, info_min=None, n=None, y2_gwas=None):
        self.ldr_gwas = ldr_gwas
        self.y2_gwas = y2_gwas
        self.snp_col = snp_col
        self.a1_col = a1_col
        self.a2_col = a2_col
        self.n_col = n_col
        self.effect_col = effect_col
        self.se_col = se_col
        self.chr_col = chr_col
        self.pos_col = pos_col
        self.z_col = z_col
        self.p_col = p_col
        self.maf_col = maf_col
        self.maf_min = maf_min
        self.info_col = info_col
        self.info_min = info_min
        self.n = n


class ProcessedArgs:
    def __init__(self, snp_col=None, a1_col=None, a2_col=None, n_col=None,
                 effect=None, null_value=None, ldr_gwas=None,
                 se_col=None, chr_col=None, pos_col=None, z_col=None,
                 p_col=None, maf_col=None, maf_min=None, info_col=None,
                 info_min=None, n=None, y2_gwas=None):
        self.ldr_gwas = ldr_gwas
        self.y2_gwas = y2_gwas
        self.snp_col = snp_col
        self.a1_col = a1_col
        self.a2_col = a2_col
        self.n_col = n_col
        self.effect = effect
        self.null_value = null_value
        self.se_col = se_col
        self.chr_col = chr_col
        self.pos_col = pos_col
        self.z_col = z_col
        self.p_col = p_col
        self.maf_col = maf_col
        self.maf_min = maf_min
        self.info_col = info_col
        self.info_min = info_min
        self.n = n

    def __eq__(self, other):
        for attr in vars(self).keys():
            if getattr(self, attr) != getattr(other, attr):
                print(
                    f"Attribute '{attr}' differs: '{getattr(self, attr)}' (self) vs '{getattr(other, attr)}' (other)")
                return False
        return True


class Test_check_input(unittest.TestCase):
    def test_good_case1(self):
        """
        a standard case for LDR gwas without MAF and INFO filtering

        """
        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    z_col='z')
        processedargs = check_input(args, log)
        true_args = ProcessedArgs(ldr_gwas=[os.path.join(MAIN_DIR, 'gwas1.txt'), os.path.join(MAIN_DIR, 'gwas2.txt')],
                                  snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                                  effect='beta', null_value=0, se_col='se', chr_col='chr', pos_col='pos',
                                  z_col='z')
        self.assertEqual(processedargs, true_args)

    def test_good_case2(self):
        """
        good cases with MAF and INFO filtering

        """
        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n=10000,
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    maf_col='maf', info_col='info')
        processedargs = check_input(args, log)
        true_args = ProcessedArgs(ldr_gwas=[os.path.join(MAIN_DIR, 'gwas1.txt'),
                                            os.path.join(MAIN_DIR, 'gwas2.txt')],
                                  snp_col='snp', a1_col='a1', a2_col='a2', n=10000,
                                  effect='beta', null_value=0, se_col='se', chr_col='chr',
                                  pos_col='pos', maf_col='maf', info_col='info',
                                  maf_min=0.01, info_min=0.9)
        self.assertEqual(processedargs, true_args)

        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n=10000,
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    maf_col='maf', info_col='info', maf_min=0.05, info_min=0.8)
        processedargs = check_input(args, log)
        true_args = ProcessedArgs(ldr_gwas=[os.path.join(MAIN_DIR, 'gwas1.txt'),
                                            os.path.join(MAIN_DIR, 'gwas2.txt')],
                                  snp_col='snp', a1_col='a1', a2_col='a2', n=10000,
                                  effect='beta', null_value=0, se_col='se', chr_col='chr',
                                  pos_col='pos', maf_col='maf', info_col='info',
                                  maf_min=0.05, info_min=0.8)
        self.assertEqual(processedargs, true_args)

        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n=10000,
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    maf_min=0.05, info_min=0.8)
        processedargs = check_input(args, log)
        true_args = ProcessedArgs(ldr_gwas=[os.path.join(MAIN_DIR, 'gwas1.txt'),
                                            os.path.join(MAIN_DIR, 'gwas2.txt')],
                                  snp_col='snp', a1_col='a1', a2_col='a2', n=10000,
                                  effect='beta', null_value=0, se_col='se', chr_col='chr',
                                  pos_col='pos')
        self.assertEqual(processedargs, true_args)

    def test_good_case3(self):
        """
        multiple ways of input for y2_gwas

        """
        # beta and se
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n', effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    )
        processedargs = check_input(args, log)
        true_args = ProcessedArgs(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                                  snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                                  effect='beta', null_value=0, se_col='se', chr_col='chr', pos_col='pos')
        self.assertEqual(processedargs, true_args)

        # or and se
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n', effect_col='or,1', se_col='se'
                    )
        processedargs = check_input(args, log)
        true_args = ProcessedArgs(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                                  snp_col='snp', a1_col='a1', a2_col='a2', se_col='se',
                                  n_col='n', effect='or', null_value=1)
        self.assertEqual(processedargs, true_args)

        # z
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n', z_col='z',
                    )
        processedargs = check_input(args, log)
        true_args = ProcessedArgs(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                                  snp_col='snp', a1_col='a1', a2_col='a2',
                                  n_col='n', z_col='z')
        self.assertEqual(processedargs, true_args)

        # p-value and effect
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n', p_col='pv', effect_col='beta,0'
                    )
        processedargs = check_input(args, log)
        true_args = ProcessedArgs(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                                  snp_col='snp', a1_col='a1', a2_col='a2',
                                  n_col='n', p_col='pv', effect='beta', null_value=0)
        self.assertEqual(processedargs, true_args)

    def test_bad_cases(self):
        # missing any required component for LDR gwas
        def exclude_key(d, key_to_exclude):
            return {key: value for key, value in d.items() if key != key_to_exclude}

        args_dict = {'ldr_gwas': os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                     'snp_col': 'snp', 'a1_col': 'a1', 'a2_col': 'a2', 'n_col': 'n',
                     'effect_col': 'beta,0', 'se_col': 'se', 'chr_col': 'chr', 'pos_col': 'pos'}
        for key in args_dict.keys():
            args_dict = exclude_key(args_dict, key)
            args = Args(**args_dict)
            with self.assertRaises(ValueError):
                check_input(args, log)

        # missing required components for y2 gwas
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n', p_col='pv')
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n')
        with self.assertRaises(ValueError):
            check_input(args, log)

        # non-existing files
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas3.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n', z_col='z')
        with self.assertRaises(FileNotFoundError):
            check_input(args, log)

        # invalid MAF/INFO threshold
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n', p_col='pv', effect_col='beta,0',
                    maf_col='maf', maf_min=-1)
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2',
                    n_col='n', p_col='pv', effect_col='beta,0',
                    info_col='maf', info_min=-1)
        with self.assertRaises(ValueError):
            check_input(args, log)


class Test_map_cols(unittest.TestCase):
    def test_map_cols(self):
        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n=10000,
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    maf_col='maf', info_col='info')
        args = check_input(args, log)
        true_col_map1 = {'SNP': 'snp', 'A1': 'a1', 'A2': 'a2',
                         'EFFECT': 'beta', 'SE': 'se', 'CHR': 'chr',
                         'POS': 'pos', 'MAF': 'maf', 'INFO': 'info',
                         'N': None, 'n': 10000, 'P': None, 'null_value': 0,
                         'Z': None, 'maf_min': 0.01, 'info_min': 0.9}
        true_col_map2 = {'snp': 'SNP', 'a1': 'A1', 'a2': 'A2',
                         'beta': 'EFFECT', 'se': 'SE', 'chr': 'CHR',
                         'pos': 'POS', 'maf': 'MAF', 'info': 'INFO'}
        col_map1, col_map2 = map_cols(args)
        self.assertEqual(true_col_map1, col_map1)
        self.assertEqual(true_col_map2, col_map2)


class Test_GWAS(unittest.TestCase):
    def get_ldr_sumstats(self, args, log, fast=False):
        args = check_input(args, log)
        cols_map1, cols_map2 = map_cols(args)
        sumstats = GWAS.from_rawdata_ldr(args.ldr_gwas,
                                         cols_map1, cols_map2,
                                         args.maf_min,
                                         args.info_min,
                                         fast)
        return sumstats

    def get_y2_sumstats(self, args, log):
        args = check_input(args, log)
        cols_map1, cols_map2 = map_cols(args)
        sumstats = GWAS.from_rawdata_y2(args.y2_gwas,
                                        cols_map1, cols_map2,
                                        args.maf_min,
                                        args.info_min)
        return sumstats

    def test_ldr_gwas(self):
        true_beta = np.array([0.0, 0.0, 0.0, 0.0]).reshape((2, 2))
        true_se = np.array([0.5, 0.5, 0.5, 0.5]).reshape((2, 2))
        true_z = None
        true_snpinfo = pd.DataFrame({'CHR': [2, 2], 'POS': [10, 4],
                                    'SNP': ['rs2', 'rs4'],
                                     'A1': ['C', 'C'],
                                     'A2': ['A', 'A'],
                                     'N': [100, 100]})
        # regular case
        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    maf_col='maf')
        sumstats = self.get_ldr_sumstats(args, log)
        # true_sumstats = GWAS(true_beta, true_se, true_z, true_snpinfo)
        # self.assertEqual(sumstats, true_sumstats)
        assert_array_equal(sumstats.beta, true_beta)
        assert_array_equal(sumstats.se, true_se)
        self.assertEqual(sumstats.z, true_z)
        assert_array_equal(sumstats.snpinfo.values, true_snpinfo.values)

        # nonexisting columns
        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    maf_col='maf11')
        with self.assertRaises(ValueError):
            self.get_ldr_sumstats(args, log, True)

        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a222', n_col='n',
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    maf_col='maf11')
        with self.assertRaises(ValueError):
            self.get_ldr_sumstats(args, log, True)

    def test_y2_gwas(self):
        true_beta = None
        true_se = None
        true_z = np.array([0.0, 0.0]).reshape((2, 1))
        true_snpinfo = pd.DataFrame({'SNP': ['rs2', 'rs4'],
                                    'A1': ['C', 'C'],
                                     'A2': ['A', 'A'],
                                     'N': [100, 100]})
        # beta and se, maf
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos',
                    maf_col='maf')
        sumstats = self.get_y2_sumstats(args, log)
        assert_array_equal(sumstats.z, true_z)
        self.assertEqual(sumstats.beta, true_beta)
        self.assertEqual(sumstats.se, true_se)
        assert_array_equal(sumstats.snpinfo.values, true_snpinfo.values)

        # or and se
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                    effect_col='or,1', se_col='se', chr_col='chr', pos_col='pos')
        sumstats = self.get_y2_sumstats(args, log)
        assert_array_equal(sumstats.z, true_z)

        # z
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                    z_col='z')
        sumstats = self.get_y2_sumstats(args, log)
        assert_array_equal(sumstats.z, true_z)

        # effect and p, info
        true_snpinfo = pd.DataFrame({'SNP': ['rs4'],
                                    'A1': ['C'],
                                     'A2': ['A'],
                                     'N': [100]})
        true_z = np.array([0.0]).reshape((1, 1))
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                    effect_col='or,1', p_col='p', info_col='info')
        sumstats = self.get_y2_sumstats(args, log)
        assert_array_equal(sumstats.z, true_z)
        assert_array_equal(sumstats.snpinfo.values, true_snpinfo.values)

        # nonexisting columns
        args = Args(y2_gwas=os.path.join(MAIN_DIR, 'gwas1.txt'),
                    snp_col='snp221', a1_col='a1', a2_col='a2', n_col='n',
                    z_col='z')
        with self.assertRaises(ValueError):
            self.get_y2_sumstats(args, log)

    def test_extract_snps(self):
        true_beta = None
        true_se = None
        true_z = np.array([0.0, 0.0]).reshape((1, 2))
        true_snpinfo = pd.DataFrame({'CHR': [2], 'POS': [10],
                                    'SNP': ['rs2'],
                                     'A1': ['C'],
                                     'A2': ['A'],
                                     'N': [100]})
        args = Args(ldr_gwas=os.path.join(MAIN_DIR, 'gwas{1:2}.txt'),
                    snp_col='snp', a1_col='a1', a2_col='a2', n_col='n',
                    effect_col='beta,0', se_col='se', chr_col='chr', pos_col='pos')
        sumstats = self.get_ldr_sumstats(args, log)
        sumstats.get_zscore()
        sumstats.extract_snps(pd.Series(['rs2'], name='SNP'))

        self.assertEqual(sumstats.beta, true_beta)
        self.assertEqual(sumstats.se, true_se)
        assert_array_equal(sumstats.z, true_z)
        assert_array_equal(sumstats.snpinfo.values, true_snpinfo.values)


class Test_read_sumstats(unittest.TestCase):
    def test_read_sumstats(self):
        true_beta = np.array([0, 0, 0, 0]).reshape((2, 2))
        true_se = np.array([1, 1, 1, 1]).reshape((2, 2))
        true_z = None
        true_snpinfo = pd.DataFrame({'CHR': [2, 2], 'POS': [10, 4],
                                    'SNP': ['rs2', 'rs4'],
                                     'A1': ['C', 'C'],
                                     'A2': ['A', 'A'],
                                     'N': [100, 100]})
        sumstats = read_sumstats('test/test_sumstats/gwas')
        assert_array_equal(sumstats.beta, true_beta)
        assert_array_equal(sumstats.se, true_se)
        self.assertEqual(sumstats.z, true_z)
        assert_array_equal(sumstats.snpinfo.values, true_snpinfo.values)

