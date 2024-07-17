import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from heig.wgs.pvalue import (
    saddle,
    _handle_invalid_pvalues
)

class Test_saddle(unittest.TestCase):
    def test_saddle(self):
        ## single voxel
        pvalues = saddle(np.array([6.0]), np.array([4.0, 3.0, 2.0, 1.0]), None)
        assert_array_almost_equal(pvalues, np.array([0.6329193]))

        ## multiple voxels
        pvalues = saddle(np.array([6.0, 5.0]), np.array([4.0, 3.0, 2.0, 1.0]), None)
        assert_array_almost_equal(pvalues, np.array([0.6329193, 0.7069172]))

        ## extreme voxels 
        pvalues = saddle(np.array([60.0]), np.array([4.0, 3.0, 2.0, 1.0]), None)
        assert_array_almost_equal(pvalues, np.array([0.0004239342]))

        pvalues = saddle(np.array([600.0]), np.array([4.0, 3.0, 2.0, 1.0]), None)
        assert_array_almost_equal(pvalues, np.array([6.531105e-34]))

        ## very small stats
        pvalues = saddle(np.array([0.1]), np.array([4.0, 3.0, 2.0, 1.0]), None)
        assert_array_almost_equal(pvalues, np.array([0.9997457]))

        pvalues = saddle(np.array([0.0001]), np.array([4.0, 3.0, 2.0, 1.0]), None)
        assert_array_almost_equal(pvalues, np.array([1.0]))

    def test_handle_invalid_pvalues(self):
        wcov_mat = np.arange(12).reshape(4,3) / 50
        wcov_mat = np.dot(wcov_mat.T, wcov_mat)
        egvalues, _ = np.linalg.eigh(wcov_mat)
        egvalues = np.flip(egvalues)
        pvalues1 = _handle_invalid_pvalues(np.array([60.0, 6.0]), wcov_mat)
        pvalues2 = saddle(np.array([60.0, 6.0]), egvalues, None)
        assert_array_almost_equal(pvalues1, pvalues2)
        

if __name__ == '__main__':
    pvalues = saddle(np.array([0.1]), np.array([400.0, 3.0, 2.0, 1.0]), None)
    saddle(np.array([600.0]), np.array([4.0, 3.0, 2.0, 1.0]), None)
    wcov_mat = np.arange(12).reshape(4,3) / 100
    wcov_mat = np.dot(wcov_mat.T, wcov_mat)
    egvalues, _ = np.linalg.eigh(wcov_mat)
    egvalues = np.flip(egvalues)
    print(_handle_invalid_pvalues(np.array([6.0]), wcov_mat))
    print(saddle(np.array([6.0]), egvalues, None))