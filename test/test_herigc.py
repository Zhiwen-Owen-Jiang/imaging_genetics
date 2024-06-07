import unittest
import pandas as pd
import numpy as np
from heig.sumstats import GWAS
from pandas.testing import assert_series_equal
from numpy.testing import assert_array_equal


from heig.herigc import (
    align_alleles,
    get_common_snps
)


class toy_GWAS:
    def __init__(self, snpinfo):
        self.snpinfo = snpinfo


class toy_LDmatrix:
    def __init__(self, ldinfo):
        self.ldinfo = ldinfo


class Test_align_alleles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snpinfo = pd.DataFrame({'SNP': ['rs1', 'rs2'],
                                'A1': ['A', 'C'],
                                'A2': ['G', 'T']})
        z = np.array([1, 2, -1, -2]).reshape(2, 2)
        cls.gwas = GWAS(None, None, z, snpinfo)

        cls.bim = pd.DataFrame({'SNP': ['rs1', 'rs2'],
                                'A1': ['A', 'C'],
                                'A2': ['G', 'T']})

    def test_matched_alleles(self):
        self.assertEqual(self.gwas, align_alleles(self.bim, self.gwas))

    def test_unmatched_alleles(self):
        snpinfo = pd.DataFrame({'SNP': ['rs1', 'rs2'],
                                'A1': ['G', 'C'],
                                'A2': ['A', 'T']})
        z = np.array([-1, -2, -1, -2]).reshape(2, 2)
        gwas = GWAS(None, None, z, snpinfo)
        self.assertEqual(self.gwas, align_alleles(self.bim, gwas))

    def test_diff_snps(self):
        bim = pd.DataFrame({'SNP': ['rs1', 'rs3'],
                            'A1': ['A', 'C'],
                            'A2': ['G', 'T']})
        with self.assertRaises(ValueError):
            align_alleles(bim, self.gwas)


class Test_get_common_snps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snpinfo = pd.DataFrame({'SNP': ['rs1', 'rs3'],
                                'A1': ['G', 'C'],
                                'A2': ['A', 'A']})
        cls.gwas = toy_GWAS(snpinfo)

        ldinfo1 = pd.DataFrame({'SNP': ['rs1', 'rs2', 'rs3'],
                                'A1': ['A', 'C', 'C'],
                                'A2': ['G', 'T', 'A']})
        cls.ld1 = toy_LDmatrix(ldinfo1)

        ldinfo2 = pd.DataFrame({'SNP': ['rs1', 'rs2', 'rs3'],
                                'A1': ['A', 'C', 'G'],
                                'A2': ['G', 'T', 'A']})
        cls.ld2 = toy_LDmatrix(ldinfo2)

        ldinfo3 = pd.DataFrame({'SNP': ['rs1', 'rs3', 'rs3'],
                                'A1': ['A', 'C', 'G'],
                                'A2': ['G', 'T', 'A']})
        cls.ld3 = toy_LDmatrix(ldinfo3)

        cls.keep_snps1 = pd.DataFrame({'SNP': ['rs2', 'rs3']})
        cls.keep_snps2 = pd.DataFrame({'SNP': ['rs4']})

    def test_null(self):
        """
        no snp list provided

        """
        with self.assertRaises(ValueError):
            get_common_snps()

    def test_case1(self):
        """
        test if snp list order matters

        """
        true_res = pd.Series({0: 'rs1', 1: 'rs3'}, name='SNP')
        assert_series_equal(true_res, get_common_snps(self.gwas, self.ld1))
        assert_series_equal(true_res, get_common_snps(self.ld1, self.gwas))

    def test_case2(self):
        """
        test different combinations of snp lists

        """
        true_res1 = pd.Series({0: 'rs3'}, name='SNP')
        true_res2 = pd.Series({0: 'rs2', 1: 'rs3'}, name='SNP')
        assert_series_equal(true_res1, get_common_snps(
            self.gwas, self.ld1, self.keep_snps1))
        assert_series_equal(true_res1, get_common_snps(
            self.gwas, self.keep_snps1))
        assert_series_equal(true_res2, get_common_snps(
            self.ld1, self.keep_snps1))

    def test_case3(self):
        """
        no common SNPs exist

        """
        with self.assertRaises(ValueError):
            get_common_snps(self.gwas, self.ld1, self.keep_snps2)

    def test_case4(self):
        """
        inconsistent alleles

        """
        true_res = pd.Series({0: 'rs1'}, name='SNP')
        assert_series_equal(true_res, get_common_snps(self.gwas, self.ld2))

    def test_case5(self):
        """
        duplicated SNPs
        
        """
        true_res = pd.Series({0: 'rs1'}, name='SNP')
        assert_series_equal(true_res, get_common_snps(self.gwas, self.ld3))


class Test_check_alleles(unittest.TestCase):
    def test_check_alleles(self):
        ldinfo1 = pd.DataFrame({'SNP': ['rs1', 'rs2', 'rs3'],
                                    'A1': ['A', 'C', 'G'],
                                    'A2': ['G', 'T', 'A']})
        ldinfo2 = pd.DataFrame({'SNP': ['rs1', 'rs2', 'rs3'],
                                    'A1': ['A', 'C', 'G'],
                                    'A2': ['G', 'T', 'A']})
        ldinfo3 = pd.DataFrame({'SNP': ['rs1', 'rs2', 'rs3'],
                                    'A1': ['A', 'C', 'G'],
                                    'A2': ['G', 'T', 'A']},
                                    index=[10,20,30])
        assert_array_equal(ldinfo1[['A1', 'A2']].values, ldinfo2[['A1', 'A2']].values)
        assert_array_equal(ldinfo1[['A1', 'A2']].values, ldinfo3[['A1', 'A2']].values)
        self.assertTrue(np.equal(ldinfo1[['A1', 'A2']].values, ldinfo3[['A1', 'A2']].values).all())



