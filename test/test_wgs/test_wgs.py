import unittest
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal


def make_sparse_banded(matrix, bandwidth):
    if not isinstance(matrix, csr_matrix):
        raise ValueError("Input matrix must be a csr_matrix.")
    
    row, col = matrix.nonzero() # both are np.array
    data = matrix.data
    
    diagonal_data = data[row == col]
    mask = (np.abs(row - col) <= bandwidth) & (col > row)
    banded_row = row[mask]
    banded_col = col[mask]
    banded_data = data[mask]
    shape = matrix.shape
    
    return diagonal_data, banded_data, banded_row, banded_col, shape


def reconstruct_matrix(diag, data, row, col, shape):
    lower_row = col
    lower_col = row
    diag_row_col = np.arange(shape[0])

    full_row = np.concatenate([row, lower_row, diag_row_col])
    full_col = np.concatenate([col, lower_col, diag_row_col])
    full_data = np.concatenate([data, data, diag])
    
    full_matrix = csr_matrix((full_data, (full_row, full_col)), shape=shape)

    return full_matrix


def sparse_banded(vset, bandwidth):
    """
    sparse upper banded matrix
    
    """
    diagonal_data = list()
    banded_data = list()
    banded_row = list()
    banded_col = list()
    n_variants = vset.shape[0]

    for start in range(0, n_variants, bandwidth):
        end1 = start + bandwidth
        end2 = end1 + bandwidth
        vset_block1 = vset[start:end1]
        vset_block2 = vset[start:end2]
        ld_rec = vset_block1 @ vset_block2.T
        ld_rec_row, ld_rec_col = ld_rec.nonzero()
        ld_rec_row += start
        ld_rec_col += start
        ld_rec_data = ld_rec.data

        diagonal_data.append(ld_rec_data[ld_rec_row == ld_rec_col])
        mask = (np.abs(ld_rec_row - ld_rec_col) <= bandwidth) & (ld_rec_col > ld_rec_row)
        banded_row.append(ld_rec_row[mask])
        banded_col.append(ld_rec_col[mask])
        banded_data.append(ld_rec_data[mask])

    diagonal_data = np.concatenate(diagonal_data)
    banded_row = np.concatenate(banded_row)
    banded_col = np.concatenate(banded_col)
    banded_data = np.concatenate(banded_data)
    shape = (n_variants, n_variants)
        
    return diagonal_data, banded_data, banded_row, banded_col, shape


class Test_sparse_matrix(unittest.TestCase):
    def test_sparse_matrix(self):
        dense_array = np.array([
            [0, 2, 1, 2],
            [3, 1, 3, 0],
            [0, 4, 0, 0],
            [1, 0, 1, 1]
        ])
        csr = csr_matrix(dense_array)
        csr = csr @ csr.T

        banded_array = np.array([
            [9, 5, 0, 0],
            [5, 19, 4, 0],
            [0, 4, 16, 0],
            [0, 0, 0, 3]
        ])

        banded = make_sparse_banded(csr, 1)
        recons = reconstruct_matrix(*banded)
        recons_array = recons.toarray()
        assert_array_equal(banded_array, recons_array)

    def test_sparse_banded(self):
        dense_array = np.array([
            [0, 2, 1, 2],
            [3, 1, 3, 0],
            [0, 4, 0, 0],
            [1, 0, 1, 1]
        ])
        csr = csr_matrix(dense_array)

        banded_array = np.array([
            [9, 5, 0, 0],
            [5, 19, 4, 0],
            [0, 4, 16, 0],
            [0, 0, 0, 3]
        ])

        banded = sparse_banded(csr, 1)
        recons = reconstruct_matrix(*banded)
        recons_array = recons.toarray()
        assert_array_equal(banded_array, recons_array)


if __name__  == '__main__':
    dense_array = np.array([
            [0, 2, 1, 2],
            [3, 1, 3, 0],
            [0, 4, 0, 0],
            [1, 0, 1, 1]
    ])
    csr = csr_matrix(dense_array)

    banded_array = np.array([
        [9, 5, 0, 0],
        [5, 19, 4, 0],
        [0, 4, 16, 0],
        [0, 0, 0, 3]
    ])

    banded = sparse_banded(csr, 1)
    recons = reconstruct_matrix(*banded)
    recons_array = recons.toarray()

