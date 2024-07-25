import unittest
import os, sys
import numpy as np
import pandas as pd
import hail as hl
from hail.linalg import BlockMatrix
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from heig.wgs.staar import (
    VariantSetTest,
    cauchy_combination
)


class Test_cauchy(unittest.TestCase):
    def test_cauchy_good_cases(self):
        # normal pvalues
        pvalues = np.array([0.5, 0.01, 0.1]).reshape(3, 1)
        assert_array_almost_equal(cauchy_combination(pvalues), np.array([0.02729619]))
        
        # extremely small pvalues
        pvalues = np.array([0.5, 0.01, 10**-17]).reshape(3, 1)
        assert_array_almost_equal(cauchy_combination(pvalues), np.array([3*10**-17]))

        # combinded, two voxels
        pvalues = np.array([0.5, 0.01, 0.1, 0.5, 0.01, 10**-17]).reshape(3, 2)
        assert_array_almost_equal(cauchy_combination(pvalues), np.array([0.02729619, 3*10**-17]))

        # add weights
        pvalues = np.array([0.5, 0.01, 0.1]).reshape(3, 1)
        weights = np.array([1, 2, 3])
        assert_array_almost_equal(cauchy_combination(pvalues, weights), np.array([0.02614868]))
        
    def test_cauchy_bad_cases(self):
        # 0 in pvalues
        pvalues = np.array([0., 0.01, 0.1, 0.5, 0.01, 10**-17]).reshape(3, 2)
        assert_array_almost_equal(cauchy_combination(pvalues), np.array([0, 3*10**-17]))

        # 1 in pvalues
        pvalues = np.array([0.5, 0.01, 0.1, 0.5, 0.01, 1]).reshape(3, 2)
        assert_array_almost_equal(cauchy_combination(pvalues), np.array([0.02729619, 1]))

        # nan in pvalues
        pvalues = np.array([np.nan, 0.01, 0.1, 0.5, 0.01, np.nan]).reshape(3, 2)
        assert_array_almost_equal(cauchy_combination(pvalues), np.array([np.nan, np.nan]))

        # nan in weights
        pvalues = np.array([0.5, 0.01, 0.1, 0.5, 0.01, 0.1]).reshape(3, 2)
        weights = np.array([1, np.nan, 3])
        assert_array_almost_equal(cauchy_combination(pvalues, weights), np.array([np.nan, np.nan]))

        # negative weights
        with self.assertRaises(ValueError):
            pvalues = np.array([0.5, 0.01, 0.1, 0.5, 0.01, 0.1]).reshape(3, 2)
            weights = np.array([1, -1, 3])
            cauchy_combination(pvalues, weights)
        
        # inconsistent length
        with self.assertRaises(ValueError):
            pvalues = np.array([0.5, 0.01, 0.1]).reshape(3, 1)
            weights = np.array([1, -1, 3])
            cauchy_combination(pvalues, weights)


class Test_VariantSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hl.init(quiet=True)
        bases = np.array([1]).reshape(1, 1) # (N, r)
        resid_ldr = np.array([-1.5, -0.5, 0.5, 1.5]).reshape(4, 1) # (n, r)
        covar = np.ones((4, 1)) # (n, p)
        cls.vset_test = VariantSetTest(bases, resid_ldr, covar)

        vset = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]).reshape(4, 3) # (n, m)
        maf = np.mean(vset, axis=0) / 2
        is_rare = np.array([False, True, True])
        vset = BlockMatrix.from_numpy(vset.T) # (m, n)
        annot_pred = np.array([1,2,3]).reshape(3,1)
        cls.vset_test.input_vset(vset, maf, is_rare, annotation_pred=annot_pred)

    def test_skat(self):
        pvalues = self.vset_test.do_inference(['V1'])
        true_pvalues = pd.DataFrame({'STAAR-O': 0.40763786107471944, 
                                     'ACAT-O': 0.48029666314262487, 
                                     'SKAT(1,25)': 0.22482785847788261, 
                                     'SKAT(1,25)-V1': 0.1935911826673326, 
                                     'STAAR-SKAT(1,25)': 0.2082148904083711, 
                                     'SKAT(1,1)': 0.30965337344050614, 
                                     'SKAT(1,1)-V1': 0.23560683333828553, 
                                     'STAAR-SKAT(1,1)': 0.26886265460031966, 
                                     'Burden(1,25)': 0.583886187905023, 
                                     'Burden(1,25)-V1': 0.432234232782132, 
                                     'STAAR-Burden(1,25)': 0.5085355733865713, 
                                     'Burden(1,1)': 0.6547208460185768, 
                                     'Burden(1,1)-V1': 0.4040862039643345, 
                                     'STAAR-Burden(1,1)': 0.5344909199210232, 
                                     'ACAT-V(1,25)': 0.5838975912961833, 
                                     'ACAT-V(1,25)-V1': 0.43265025772359217, 
                                     'STAAR-ACAT-V(1,25)': 0.508759196328998, 
                                     'ACAT-V(1,1)': 0.6247893996804713, 
                                     'ACAT-V(1,1)-V1': 0.5195962989957652, 
                                     'STAAR-ACAT-V(1,1)': 0.5742355066368832},
                                     index=[0])
        assert_frame_equal(pvalues, true_pvalues)


if __name__ == '__main__':
    # pvalues = np.array([0, 0.01, 0.1, 0.5, 0.01, 10**-17]).reshape(3, 2)
    # cauchy_combination(pvalues)

    bases = np.array([1]).reshape(1, 1) # (1, r)
    resid_ldr = np.array([-1.5, -0.5, 0.5, 1.5]).reshape(4, 1) # (n, r)
    covar = np.ones((4, 1)) # (n, p)
    vset_test = VariantSetTest(bases, resid_ldr, covar)

    vset = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]).reshape(4, 3) # (n, m)
    maf = np.mean(vset, axis=0) / 2
    is_rare = np.array([False, True, True])
    vset = BlockMatrix.from_numpy(vset.T) # (m, n)
    annot_pred = np.array([1,2,3]).reshape(3,1)
    vset_test.input_vset(vset, maf, is_rare, annotation_pred=annot_pred)
    pvalues = vset_test.do_inference(['V1'])
    pvalues

    # bases = np.array([1]).reshape(1, 1)
    # resid_ldr = np.load('resid_ldr.npy')
    # covar = np.load('covar.npy')
    # vset_test = VariantSetTest(bases, resid_ldr, covar)
    # vset = np.load('vset_array.npy') # (m, n)
    # annot_pred = np.load('phred_cate.npy')
    # maf = np.mean(vset, axis=1) / 2
    # is_rare = np.sum(vset, axis=1) <= 2
    # vset = BlockMatrix.from_numpy(vset) # (m, n)
    # vset_test.input_vset(vset, maf, is_rare, annotation_pred=annot_pred)
    # pvalues = vset_test.do_inference(["CADD",
    #                "LINSIGHT",
    #                "FATHMM.XF",
    #                "aPC.EpigeneticActive",
    #                "aPC.EpigeneticRepressed",
    #                "aPC.EpigeneticTranscription",
    #                "aPC.Conservation",
    #                "aPC.LocalDiversity",
    #                "aPC.LocalDiversity(-)",
    #                "aPC.Mappability",
    #                "aPC.TF",
    #                "aPC.Protein"
    #                ])
