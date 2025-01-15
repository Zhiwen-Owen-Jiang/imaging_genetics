import os
import logging
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal

from heig.ldmatrix import (
    partition_genome,
    find_loc,
    get_sub_blocks,
    LDmatrix,
    LDmatrixBED,
    check_input,
)

MAIN_DIR = os.path.join(os.getcwd(), "test", "test_ldmatrix")
log = logging.getLogger()
log.setLevel(logging.INFO)


class Test_LDmatrix(unittest.TestCase):
    def setUp(self):
        self.ld1 = LDmatrix(os.path.join(MAIN_DIR, "ld1_chr{21:22}"))
        self.ld2 = LDmatrix(os.path.join(MAIN_DIR, "ld1_chr21"))

    def test_good_case1(self):
        ldinfo = pd.DataFrame(
            {
                "CHR": [21, 21, 22, 22],
                "SNP": ["rs1148733", "rs2747339", "rs62224618", "rs116911124"],
                "CM": [0, 0, 0, 0],
                "POS": [14677301, 14748916, 16057417, 16495833],
                "A1": ["C", "T", "T", "A"],
                "A2": ["T", "C", "C", "C"],
                "MAF": [0.1, 0.2, 0.1, 0.2],
                "block_idx": [0, 0, 1, 1],
                "block_idx2": [0, 1, 0, 1],
                "ldscore": [1.592, 1.342, 1.296, 1.285],
            }
        )
        block_sizes = [2, 2]
        block_ranges = [(0, 2), (2, 4)]
        data = np.array([1, 1, 1, 1]).reshape(2, 2)

        assert_frame_equal(self.ld1.ldinfo, ldinfo)
        self.assertEqual(self.ld1.block_sizes, block_sizes)
        self.assertEqual(self.ld1.block_ranges, block_ranges)
        for data_ in self.ld1.data:
            assert_array_equal(data_, data)

    def test_good_case2(self):
        ldinfo = pd.DataFrame(
            {
                "CHR": [21, 21],
                "SNP": ["rs1148733", "rs2747339"],
                "CM": [0, 0],
                "POS": [14677301, 14748916],
                "A1": ["C", "T"],
                "A2": ["T", "C"],
                "MAF": [0.1, 0.2],
                "block_idx": [0, 0],
                "block_idx2": [0, 1],
                "ldscore": [1.592, 1.342],
            }
        )
        block_sizes = [2]
        block_ranges = [(0, 2)]
        data = np.array([1, 1, 1, 1]).reshape(2, 2)

        assert_frame_equal(self.ld2.ldinfo, ldinfo)
        self.assertEqual(self.ld2.block_sizes, block_sizes)
        self.assertEqual(self.ld2.block_ranges, block_ranges)
        for data_ in self.ld2.data:
            assert_array_equal(data_, data)

    def test_bad_case(self):
        with self.assertRaises(ValueError):
            LDmatrix(os.path.join(MAIN_DIR, "ld1_chr21_dup"))
        with self.assertRaises(ValueError):
            self.ld1._merge_ldinfo([])

    def test_extract(self):
        self.ld2.extract(["rs1148733"])
        ldinfo = pd.DataFrame(
            {
                "CHR": [21],
                "SNP": ["rs1148733"],
                "CM": [0],
                "POS": [14677301],
                "A1": ["C"],
                "A2": ["T"],
                "MAF": [0.1],
                "block_idx": [0],
                "block_idx2": [0],
                "ldscore": [1.592],
            }
        )
        block_sizes = [1]
        block_ranges = [(0, 1)]
        data = np.array([1, 1]).reshape(1, 2)

        assert_frame_equal(self.ld2.ldinfo, ldinfo)
        self.assertEqual(self.ld2.block_sizes, block_sizes)
        self.assertEqual(self.ld2.block_ranges, block_ranges)
        for data_ in self.ld2.data:
            assert_array_equal(data_, data)

    def test_merge_blocks(self):
        self.assertEqual([tuple([0]), tuple([1])], self.ld1.merge_blocks())


class Test_find_loc(unittest.TestCase):
    def test_find_loc(self):
        self.assertEqual(2, find_loc([1, 3, 5, 7], 6))
        self.assertEqual(2, find_loc([1, 3, 5, 7], 5))
        self.assertEqual(3, find_loc([1, 3, 5, 7], 8))
        self.assertEqual(-1, find_loc([1, 3, 5, 7], -1))


class Test_get_sub_blocks(unittest.TestCase):
    def test_get_sub_blocks(self):
        self.assertEqual([(0, 1000), (1000, 2000)], get_sub_blocks(0, 2000))
        self.assertEqual([(0, 1250), (1250, 2500)], get_sub_blocks(0, 2500))
        self.assertEqual([(0, 1000), (1000, 2001)], get_sub_blocks(0, 2001))
        self.assertEqual([(0, 1001), (1001, 2002)], get_sub_blocks(0, 2002))
        self.assertEqual(
            [(0, 1000), (1000, 2000), (2000, 3000)], get_sub_blocks(0, 3000)
        )


class Test_partition_genome(unittest.TestCase):
    def test_partition_genome_case1(self):
        """
        regular case

        """
        ldinfo = pd.DataFrame(
            {"CHR": [1, 1, 1, 1, 1], "POS": [100, 200, 300, 400, 500]}
        )
        part = pd.DataFrame(
            {
                0: [1, 1, 1, 1, 1],
                1: [100, 200, 300, 400, 500],
                2: [200, 300, 400, 500, 600],
            }
        )
        true_num_snps_part = [2, 1, 1, 1]
        true_ld_bim = pd.DataFrame(
            {
                "CHR": [1, 1, 1, 1, 1],
                "POS": [100, 200, 300, 400, 500],
                "block_idx": [0, 0, 1, 2, 3],
                "block_idx2": [0, 1, 0, 0, 0],
            }
        )
        num_snps_part, ld_bim = partition_genome(ldinfo, part, log)
        self.assertEqual(true_num_snps_part, num_snps_part)
        assert_frame_equal(true_ld_bim, ld_bim)

    def test_partition_genome_case2(self):
        """
        an empty block

        """
        ldinfo = pd.DataFrame({"CHR": [1, 1, 1, 1], "POS": [100, 200, 300, 400]})
        part = pd.DataFrame(
            {0: [1, 1, 1, 1], 1: [100, 200, 300, 500], 2: [200, 300, 500, 600]}
        )
        true_num_snps_part = [2, 1, 1]
        true_ld_bim = pd.DataFrame(
            {
                "CHR": [1, 1, 1, 1],
                "POS": [100, 200, 300, 400],
                "block_idx": [0, 0, 1, 2],
                "block_idx2": [0, 1, 0, 0],
            }
        )
        num_snps_part, ld_bim = partition_genome(ldinfo, part, log)
        self.assertEqual(true_num_snps_part, num_snps_part)
        assert_frame_equal(true_ld_bim, ld_bim)

    def test_partition_genome_case3(self):
        """
        all empty blocks

        """
        ldinfo = pd.DataFrame({"CHR": [1, 1, 1, 1], "POS": [100, 200, 300, 400]})
        part1 = pd.DataFrame(
            {0: [1, 1, 1, 1], 1: [1000, 2000, 3000, 5000], 2: [2000, 3000, 5000, 6000]}
        )
        part2 = pd.DataFrame(
            {0: [1, 1, 1, 1], 1: [10, 20, 30, 50], 2: [20, 30, 50, 60]}
        )
        with self.assertRaises(ValueError):
            partition_genome(ldinfo, part1, log)
        with self.assertRaises(ValueError):
            partition_genome(ldinfo, part2, log)

    def test_partition_genome_case4(self):
        """
        large blocks

        """
        ldinfo = pd.DataFrame({"CHR": [1] * 3000, "POS": list(range(0, 6000, 2))})
        part = pd.DataFrame({0: [1, 1], 1: [0, 5000], 2: [5000, 7000]})
        true_num_snps_part = [1250, 1251, 499]
        true_ld_bim = pd.DataFrame(
            {
                "CHR": [1] * 3000,
                "POS": list(range(0, 6000, 2)),
                "block_idx": [0] * 1250 + [1] * 1251 + [2] * 499,
                "block_idx2": list(range(1250)) + list(range(1251)) + list(range(499)),
            }
        )
        num_snps_part, ld_bim = partition_genome(ldinfo, part, log)
        self.assertEqual(true_num_snps_part, num_snps_part)
        assert_frame_equal(true_ld_bim, ld_bim)


class Test_LDmatrixBED(unittest.TestCase):
    def test_one_snp(self):
        def snp_getter(n):
            data = np.array([1, 2, 1, 1, 1]).reshape(5, 1)
            return data[:, :n]

        num_snps_part = [1]
        ldinfo = pd.DataFrame(
            {
                "CHR": [21],
                "SNP": ["rs1148733"],
                "CM": [0],
                "POS": [14677301],
                "A1": ["C"],
                "A2": ["T"],
                "MAF": [0.1],
                "block_idx": [0],
                "block_idx2": [0],
            }
        )
        prop = 0.8
        true_ldinfo = pd.DataFrame(
            {
                "CHR": [21],
                "SNP": ["rs1148733"],
                "CM": [0],
                "POS": [14677301],
                "A1": ["C"],
                "A2": ["T"],
                "MAF": [0.1],
                "block_idx": [0],
                "block_idx2": [0],
                "ldscore": [1.0],
            }
        )

        ld = LDmatrixBED(num_snps_part, ldinfo, snp_getter, prop, inv=False)
        self.assertEqual([np.array([[1]], dtype=np.float64)], ld.data)
        assert_frame_equal(true_ldinfo, ld.ldinfo)


class Args:
    def __init__(self, bfile, partition, ld_regu, maf_min):
        self.bfile = bfile
        self.partition = partition
        self.ld_regu = ld_regu
        self.maf_min = maf_min


class Test_check_input(unittest.TestCase):
    def test_case1(self):
        """
        good case

        """
        args = Args(
            bfile=f"{os.path.join(MAIN_DIR, 'plink')},{os.path.join(MAIN_DIR, 'plink1')}",
            partition=os.path.join(MAIN_DIR, "genome_part.txt"),
            ld_regu="0.9,0.8",
            maf_min=0.01,
        )
        ld_bfile, ld_inv_bfile, ld_regu, ld_inv_regu = check_input(args)
        self.assertEqual(os.path.join(MAIN_DIR, "plink"), ld_bfile)
        self.assertEqual(os.path.join(MAIN_DIR, "plink1"), ld_inv_bfile)
        self.assertEqual(0.9, ld_regu)
        self.assertEqual(0.8, ld_inv_regu)

    def test_case2(self):
        """
        bad case: wrong separator

        """
        args = Args(
            bfile=f"{os.path.join(MAIN_DIR, 'plink')};{os.path.join(MAIN_DIR, 'plink1')}",
            partition=os.path.join(MAIN_DIR, "genome_part.txt"),
            ld_regu="0.9,0.8",
            maf_min=0.01,
        )
        with self.assertRaises(ValueError):
            check_input(args)

        args = Args(
            bfile=f"{os.path.join(MAIN_DIR, 'plink')},{os.path.join(MAIN_DIR, 'plink1')}",
            partition=os.path.join(MAIN_DIR, "genome_part.txt"),
            ld_regu="0.9;0.8",
            maf_min=0.01,
        )
        with self.assertRaises(ValueError):
            check_input(args)

    def test_case3(self):
        """
        bad case: wrong LD regularization/maf_min

        """
        args = Args(
            bfile=f"{os.path.join(MAIN_DIR, 'plink')};{os.path.join(MAIN_DIR, 'plink1')}",
            partition=os.path.join(MAIN_DIR, "genome_part.txt"),
            ld_regu="1,0.8",
            maf_min=0.01,
        )
        with self.assertRaises(ValueError):
            check_input(args)

        args = Args(
            bfile=f"{os.path.join(MAIN_DIR, 'plink')};{os.path.join(MAIN_DIR, 'plink1')}",
            partition=os.path.join(MAIN_DIR, "genome_part.txt"),
            ld_regu="0.9,0.8",
            maf_min=0.5,
        )
        with self.assertRaises(ValueError):
            check_input(args)

    def test_case4(self):
        """
        bad case: lack of required parameters

        """
        args = Args(
            bfile=f"{os.path.join(MAIN_DIR, 'plink')};{os.path.join(MAIN_DIR, 'plink1')}",
            partition=os.path.join(MAIN_DIR, "genome_part.txt"),
            ld_regu=None,
            maf_min=None,
        )
        with self.assertRaises(ValueError):
            check_input(args)

        args = Args(
            bfile=None,
            partition=os.path.join(MAIN_DIR, "genome_part.txt"),
            ld_regu="0.9,0.8",
            maf_min=None,
        )
        with self.assertRaises(ValueError):
            check_input(args)

        args = Args(
            bfile=f"{os.path.join(MAIN_DIR, 'plink')};{os.path.join(MAIN_DIR, 'plink1')}",
            partition=None,
            ld_regu="0.9,0.8",
            maf_min=None,
        )
        with self.assertRaises(ValueError):
            check_input(args)
