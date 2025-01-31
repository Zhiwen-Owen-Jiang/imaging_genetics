import unittest
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal


def find_loc(num_list, target):
    l = 0
    r = len(num_list) - 1
    while l <= r:
        mid = (l + r) // 2
        if num_list[mid] > target:
            r = mid - 1
        else:
            l = mid + 1
    return l

def get_window_size(demo, bandwidth):
    window_size = list()
    for start_idx, start in enumerate(demo):
        end_idx = find_loc(demo, start + bandwidth)
        window_size.append(end_idx - start_idx)
    window_size.append(0)
    return window_size

def get_blocks(window_size):
    blocks = list()
    start = 0
    for i in range(len(window_size) - 1):
        if i == len(window_size) - 2 or window_size[i] <= window_size[i+1]:
            blocks.append((start, i - start + 1, window_size[start]))
            start = i + 1
    return blocks


def sparse_banded(vset, window_size):
    """
    Create a sparse banded LD matrix by blocks

    1. Computing a LD rectangle of shape (bandwidth, 2bandwidth)
    2. Extracting the upper band of bandwidth
    3. Moving to the next rectangle

    Parameters:
    ------------
    vset (m, n): csr_matrix
    window_size (list): the ith element is the number of variants in a window 
        starting with the ith variant; the last element must be 0

    Returns:
    ---------
    banded_matrix: Sparse banded matrix.

    """
    diagonal_data = list()
    banded_data = list()
    banded_row = list()
    banded_col = list()
    n_variants = vset.shape[0]

    # window_size = get_window_size(position, bandwidth)
    blocks = get_blocks(window_size)
    # start = 0

    for block in blocks:
        start = block[0]
        end1 = start + block[1]
        end2 = start + block[2]
        vset_block1 = vset[start:end1].astype(np.uint16)
        vset_block2 = vset[start:end2].astype(np.uint16)
        ld_rec = vset_block1 @ vset_block2.T
        ld_rec_row, ld_rec_col = ld_rec.nonzero()
        ld_rec_row += start
        ld_rec_col += start
        ld_rec_data = ld_rec.data
        # start = end1

        diagonal_data.append(ld_rec_data[ld_rec_row == ld_rec_col])
        mask = ld_rec_col > ld_rec_row
        
        banded_row.append(ld_rec_row[mask])
        banded_col.append(ld_rec_col[mask])
        banded_data.append(ld_rec_data[mask])

    diagonal_data = np.concatenate(diagonal_data)
    banded_row = np.concatenate(banded_row)
    banded_col = np.concatenate(banded_col)
    banded_data = np.concatenate(banded_data)
    shape = np.array([n_variants, n_variants])
    
    if len(diagonal_data) != n_variants:
        raise ValueError('0 is not allowed in the diagonal of LD matrix')

    return diagonal_data, banded_data, banded_row, banded_col, shape


def reconstruct_vset_ld(diag, data, row, col, shape):
    """
    Reconstructing a sparse matrix.

    Parameters:
    ------------
    diag: diagonal data
    data: banded data
    row: row index of upper band
    col: col index of lower band
    shape: sparse matrix shape

    Returns:
    ---------
    full_matrix: sparse banded matrix.

    """
    lower_row = col
    lower_col = row
    diag_row_col = np.arange(shape[0])

    full_row = np.concatenate([row, lower_row, diag_row_col])
    full_col = np.concatenate([col, lower_col, diag_row_col])
    full_data = np.concatenate([data, data, diag])

    full_matrix = csr_matrix((full_data, (full_row, full_col)), shape=shape)

    return full_matrix


def sparse_banded_naive(vset, window_size):
    n_variants = vset.shape[0]
    full_matrix = np.zeros((n_variants, n_variants))
    for i, window in enumerate(window_size):
        block = vset[i:i+window]
        block = block.astype(np.uint16)
        full_matrix[i:i+window, i:i+window] = (block @ block.T).toarray()

    return full_matrix


class Test_sparse_matrix(unittest.TestCase):
    def test_sparse_matrix(self):
        vset = csr_matrix(np.ones((8, 5)))
        
        window_size = [4,3,3,5,4,3,2,1,0]
        full_matrix = reconstruct_vset_ld(*sparse_banded(vset, window_size)).toarray()
        target_matrix = sparse_banded_naive(vset, window_size)
        assert_array_equal(full_matrix, target_matrix)

        window_size = [5,5,5,5,4,3,2,1,0]
        full_matrix = reconstruct_vset_ld(*sparse_banded(vset, window_size)).toarray()
        target_matrix = sparse_banded_naive(vset, window_size)
        assert_array_equal(full_matrix, target_matrix)


if __name__ == '__main__':
    import hail as hl
    from scipy.sparse import load_npz

    locus_chr21_phase456 = hl.read_table('/work/users/o/w/owenjf/image_genetics/methods/real_data_analysis/wes/sparse_genotype_phase456/ukb23150_c21_b0_v1_phase456_hwe_maf_qc_locus_info.ht')
    vset_chr21_phase456 = load_npz('/work/users/o/w/owenjf/image_genetics/methods/real_data_analysis/wes/sparse_genotype_phase456/ukb23150_c21_b0_v1_phase456_hwe_maf_qc_genotype.npz')
    position_chr21_phase456 = locus_chr21_phase456.locus.position.collect()
    position_chr21_phase456 = np.array(position_chr21_phase456)
    
    vset_chr21_phase456_ = vset_chr21_phase456[7000: 15000]
    position_chr21_phase456_ = position_chr21_phase456[7000: 15000]

    window_size_chr21_phase456 = get_window_size(position_chr21_phase456_, bandwidth=1245)
    blocks_chr21_phase456 = get_blocks(window_size_chr21_phase456)
    len(blocks_chr21_phase456)

    full_matrix = reconstruct_vset_ld(*sparse_banded(vset_chr21_phase456_, window_size_chr21_phase456)).toarray()
    target_matrix = sparse_banded_naive(vset_chr21_phase456_, window_size_chr21_phase456)

    print(np.equal(full_matrix, target_matrix).all())