import unittest
import logging

from heig.fpca import (
    get_n_top,
    get_batch_size
)

log = logging.getLogger()
log.setLevel(logging.INFO)


class Test_get_n_top(unittest.TestCase):
    def test_get_n_top(self):
        self.assertEqual(10, get_n_top(max_n_pc=10, dim=1, n_sub=100,
                         all=False, n_ldrs=15, log=log))
        self.assertEqual(10, get_n_top(max_n_pc=10, dim=3, n_sub=100,
                         all=False, n_ldrs=15, log=log))
        self.assertEqual(15, get_n_top(max_n_pc=20, dim=3, n_sub=100,
                         all=False, n_ldrs=15, log=log))
        self.assertEqual(10, get_n_top(max_n_pc=20, dim=3, n_sub=100,
                         all=False, n_ldrs=None, log=log))
        self.assertEqual(10, get_n_top(max_n_pc=20, dim=3, n_sub=10,
                         all=True, n_ldrs=None, log=log))

class Test_get_batch_size(unittest.TestCase):
    def test_get_batch_size(self):
        self.assertEqual(20000, get_batch_size(n_top=10001, n_sub=20000, n_voxels=15001))
        self.assertEqual(50000, get_batch_size(n_top=10001, n_sub=50000, n_voxels=15001))
        self.assertEqual(30000, get_batch_size(n_top=10001, n_sub=60000, n_voxels=15001))
        self.assertEqual(50000, get_batch_size(n_top=10001, n_sub=100000, n_voxels=15001))
        self.assertEqual(50000, get_batch_size(n_top=4000, n_sub=100000, n_voxels=15001))
        self.assertEqual(10000, get_batch_size(n_top=8000, n_sub=10000, n_voxels=15001))
        self.assertEqual(10000, get_batch_size(n_top=2000, n_sub=10000, n_voxels=15001))
        self.assertEqual(5000, get_batch_size(n_top=5000, n_sub=5000, n_voxels=150000))
        self.assertEqual(10000, get_batch_size(n_top=2000, n_sub=10000, n_voxels=2000))
        self.assertEqual(10000, get_batch_size(n_top=10000, n_sub=10000, n_voxels=2000))
        self.assertEqual(1000, get_batch_size(n_top=1000, n_sub=1000, n_voxels=2000))