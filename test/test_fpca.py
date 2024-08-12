import unittest
import logging
import numpy as np

from heig.fpca import (
    fPCA
)

log = logging.getLogger()
log.setLevel(logging.INFO)


def get_batch_size(n_top, max_n_pc, n_sub):
    """
    Adaptively determine batch size

    Parameters:
    ------------
    max_n_pc: the maximum possible number of components
    n_sub: the sample size

    Returns:
    ---------
    batch size for IncrementalPCA

    """
    if max_n_pc <= 15000:
        if n_sub <= 50000:
            return n_sub
        else:
            return n_sub // (n_sub // 50000 + 1)
    else:
        if n_top > 15000 or n_sub > 50000:
            i = 2
            while n_sub // i > 50000:
                i += 1
            return n_sub // i
        else:
            return n_sub


class Test_get_n_top(unittest.TestCase):
    def test_get_n_top(self):
        fpca = fPCA(n_sub=100, max_n_pc=10, dim=1, compute_all=False, n_ldrs=15)
        self.assertEqual(10, fpca.n_top)

        fpca = fPCA(n_sub=100, max_n_pc=10, dim=3, compute_all=False, n_ldrs=15)
        self.assertEqual(10, fpca.n_top)

        fpca = fPCA(n_sub=100, max_n_pc=20, dim=3, compute_all=False, n_ldrs=15)
        self.assertEqual(15, fpca.n_top)

        fpca = fPCA(n_sub=100, max_n_pc=20, dim=3, compute_all=False, n_ldrs=None)
        self.assertEqual(10, fpca.n_top)

        fpca = fPCA(n_sub=10, max_n_pc=20, dim=3, compute_all=True, n_ldrs=None)
        self.assertEqual(10, fpca.n_top)

class Test_get_batch_size(unittest.TestCase):
    def test_get_batch_size(self):
        self.assertEqual(20000, get_batch_size(n_top=10001, n_sub=20000, max_n_pc=15001))
        self.assertEqual(50000, get_batch_size(n_top=10001, n_sub=50000, max_n_pc=15001))
        self.assertEqual(30000, get_batch_size(n_top=10001, n_sub=60000, max_n_pc=15001))
        self.assertEqual(50000, get_batch_size(n_top=10001, n_sub=100000, max_n_pc=15001))
        self.assertEqual(50000, get_batch_size(n_top=4000, n_sub=100000, max_n_pc=15001))
        self.assertEqual(10000, get_batch_size(n_top=8000, n_sub=10000, max_n_pc=10000))
        self.assertEqual(10000, get_batch_size(n_top=2000, n_sub=10000, max_n_pc=10000))
        self.assertEqual(5000, get_batch_size(n_top=5000, n_sub=5000, max_n_pc=5000))
        self.assertEqual(10000, get_batch_size(n_top=2000, n_sub=10000, max_n_pc=2000))
        self.assertEqual(10000, get_batch_size(n_top=10000, n_sub=10000, max_n_pc=2000))
        self.assertEqual(1000, get_batch_size(n_top=1000, n_sub=1000, max_n_pc=1000))