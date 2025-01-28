import unittest
import os, sys
import numpy as np
from scipy.sparse import lil_matrix
from numpy.testing import assert_array_equal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from heig.wgs.mt import flip_variants


class Test_flip_variants(unittest.TestCase):
    def test_good_cases(self):
        target_array = np.array(
            [[1, 0, 0, 0],
             [0, 0, 2, 1],
             [0, 0, 0, 1]],
             dtype=np.int8
        )
        to_flip_array = np.array(
            [[1, 2, 2, 2],
             [0, 0, 2, 1],
             [0, 0, 0, 1]],
             dtype=np.int8
        )
        to_flip_matrix = lil_matrix(to_flip_array)
        flipped_matrix = flip_variants(to_flip_matrix)
        assert_array_equal(target_array, flipped_matrix.toarray())

        to_flip_array = np.array(
            [[1, 0, 0, 0],
             [2, 2, 0, 1],
             [2, 2, 2, 1]],
             dtype=np.int8
        )
        to_flip_matrix = lil_matrix(to_flip_array)
        flipped_matrix = flip_variants(to_flip_matrix)
        assert_array_equal(target_array, flipped_matrix.toarray())