import unittest
import numpy as np
from numpy.testing import assert_array_equal
import heig.input.dataset as ds


def check_accepted_args(module, args):
    """
    a similar version for easy test

    """
    accepted_args = {
        "heri_gc": {
            "out",
            "ld_inv",
            "ld",
            "y2_sumstats",
            "overlap",
            "heri_only",
            "n_ldrs",
            "ldr_sumstats",
            "bases",
            "inner_ldr",
            "extract",
        },
        "kernel_smooth": {
            "out",
            "keep",
            "image_dir",
            "image_suffix",
            "surface_mesh",
            "bw_opt",
        },
        "fpca": {
            "out",
            "image",
            "sm_image",
            "prop",
            "all",
            "n_ldrs",
            "keep",
            "covar",
            "cat_covar_list",
        },
        "ld_matrix": {
            "out",
            "partition",
            "ld_regu",
            "bfile",
            "keep",
            "extract",
            "maf_min",
        },
        "sumstats": {
            "out",
            "ldr_gwas",
            "y2_gwas",
            "n",
            "n_col",
            "chr_col",
            "pos_col",
            "snp_col",
            "a1_col",
            "a2_col",
            "effect_col",
            "se_col",
            "z_col",
            "p_col",
            "maf_col",
            "maf_min",
            "info_col",
            "info_min",
            "fast_sumstats",
        },
        "voxel_gwas": {
            "out",
            "sig_thresh",
            "voxel",
            "range",
            "extract",
            "ldr_sumstats",
            "n_ldrs",
            "inner_ldr",
            "bases",
        },
        "gwas": {
            "out",
            "ldrs",
            "grch37",
            "threads",
            "mem",
            "geno_mt" "covar",
            "cat_covar_list",
            "bfile",
        },
    }

    ignored_args = []
    for k, v in args.items():
        if v is None or not v:
            continue
        elif k not in accepted_args[module]:
            ignored_args.append(k)

    if len(ignored_args) > 0:
        ignored_args = [f"--{arg.replace('_', '-')}" for arg in ignored_args]
        ignored_args_str = ", ".join(ignored_args)
        print(
            f"WARNING: {ignored_args_str} are ignored by --{module.replace('_', '-')}."
        )

    return ignored_args


class Test_check_accepted_args(unittest.TestCase):
    def test_check_accepted_args(self):
        args = {"non_exist1": "aaa", "non_exist2": "bbb", "out": "ccc"}
        true_ignored_args = ["--non-exist1", "--non-exist2"]

        ignored_args = check_accepted_args("heri_gc", args)
        self.assertEqual(true_ignored_args, ignored_args)


def process_args(args):
    try:
        args.voxel = np.array([int(voxel) - 1 for voxel in ds.parse_input(args.voxel)])
    except ValueError:
        ds.check_existence(args.voxel[0])
        args.voxel = ds.read_voxel(args.voxel[0])
    if np.min(args.voxel) <= -1:
        raise ValueError("voxel index must be one-based")


class Args:
    def __init__(self, voxel):
        self.voxel = voxel


class Test_process_args(unittest.TestCase):
    def test_voxel(self):
        args = Args(voxel="3")
        process_args(args)
        assert_array_equal(np.array([2]), args.voxel)

        args = Args(voxel="{2:3}")
        process_args(args)
        assert_array_equal(np.array([1, 2]), args.voxel)

        args = Args(voxel="2,3")
        process_args(args)
        assert_array_equal(np.array([1, 2]), args.voxel)

        args = Args(voxel="abc")
        with self.assertRaises(FileNotFoundError):
            process_args(args)
