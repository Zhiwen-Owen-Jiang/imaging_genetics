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
        bases = np.array([1, 2, 3]).reshape(3, 1) # (N, r)
        resid_ldr = np.array([-1.5, -0.5, 0.5, 1.5]).reshape(4, 1) # (n, r)
        covar = np.ones((4, 1)) # (n, p)
        cls.vset_test = VariantSetTest(bases, resid_ldr, covar)

        vset = np.array([1, 2, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]).reshape(4, 3) # (n, m)
        maf = np.mean(vset, axis=0) / 2
        is_rare = np.array([False, False, True])
        vset = BlockMatrix.from_numpy(vset.T) # (m, n)
        cls.vset_test.input_vset(vset, maf, is_rare, annotation_pred=None)

    def test_skat(self):
        pvalues = self.vset_test.do_inference()
        true_pvalues = pd.DataFrame({'STAAR-O': [0.18508660335357557, 0.18508660335357557, 0.18508660335357557],
                                     'ACAT-O': [0.18508660335357557, 0.18508660335357557, 0.18508660335357557],
                                     'SKAT(1,25)': [0.12389301828489946, 0.12389301828489946, 0.12389301828489946],
                                     'SKAT(1,1)': [0.23087030555777366, 0.23087030555777366, 0.23087030555777366],
                                     'Burden(1,25)': [0.1237858141574412, 0.1237858141574412, 0.12378581415744118],
                                     'Burden(1,1)': [0.5838824207703648, 0.5838824207703648, 0.583882420770365],
                                     'ACAT-V(1,25)': [0.12137422208635251, 0.12137422208635251, 0.12137422208635251],
                                     'ACAT-V(1,1)': [0.2671592334810178, 0.2671592334810178, 0.2671592334810178]})
        assert_frame_equal(pvalues, true_pvalues)


if __name__ == '__main__':
    pvalues = np.array([0, 0.01, 0.1, 0.5, 0.01, 10**-17]).reshape(3, 2)
    cauchy_combination(pvalues)

    bases = np.array([1, 2, 3]).reshape(3, 1) # (N, r)
    resid_ldr = np.array([-1.5, -0.5, 0.5, 1.5]).reshape(4, 1) # (n, r)
    covar = np.ones((4, 1)) # (n, p)
    vset_test = VariantSetTest(bases, resid_ldr, covar)

    vset = np.array([1, 2, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]).reshape(4, 3) # (n, m)
    maf = np.mean(vset, axis=0) / 2
    is_rare = np.array([False, False, True])
    vset = BlockMatrix.from_numpy(vset.T, block_size=2048) # (m, n)
    vset_test.input_vset(vset, maf, is_rare, annotation_pred=None)
    pvalues = vset_test.do_inference()
    pvalues
