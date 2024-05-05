import os
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold

import input.dataset as ds
import input.genotype as gt
from ldmatrix import partition_genome


"""
TODO: support parallel

"""


class Relatedness:
    """
    Remove relatedness by ridge regression.
    Level 0 ridge:
    1. Read SNPs by LD block
    2. Generate a range of shrinkage parameters
    For each LDR:
    3. Compute predictors for each pair of LD block and shrinkage parameter
    4. Save the predictors (?)

    Level 1 ridge:
    For each LDR:
    1. Generate a range of shrinkage parameters
    2. Cross-validation to select the optimal shrinkage parameter
    2. Compute predictors using the optimal parameter
    3. Save the predictors by each chromosome

    """

    def __init__(self, n_snps, ldrs, covar):
        """
        num_snps_part: a dict of chromosome: [LD block sizes]
        snp_getter: a generator for getting SNPs
        n_snps: a positive number of total array snps
        ldrs: n by r matrix
        covar: n by p matrix (preprocessed, including the intercept)

        """
        shrinkage = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
        self.shrinkage = n_snps * (1 - shrinkage) / shrinkage
        self.kf = KFold(n_splits=len(self.shrinkage))
        self.n, self.r = ldrs.shape

        self.inner_covar_inv = np.linalg.inv(np.dot(covar.T, covar))
        self.resid_ldrs = ldrs - \
            np.dot(np.dot(covar, self.inner_covar_inv), np.dot(covar.T, ldrs))
        self.covar = covar

    def level0_ridge_block(self, block):
        """
        Compute level 0 ridge prediction for a genotype block.
        Missing values in each block assumed to be imputed by 0.

        Parameters:
        ------------
        block: n by m matrix of a genotype block

        Returns:
        ---------
        level0_preds: a (r by n by 5) array of predictions

        """
        level0_preds = np.zeros((self.r, self.n, len(self.shrinkage)))
        block_covar = np.dot(block.T, self.covar)  # m by p
        resid_block = block - \
            np.dot(np.dot(self.covar, self.inner_covar_inv),
                   block_covar.T)  # n by m
        proj_inner_block = np.dot(block.T, resid_block)  # m by m
        proj_block_ldrs = np.dot(block.T, self.resid_ldrs)  # m by r
        for i, param in enumerate(self.shrinkage):
            preds = np.dot(
                resid_block,
                np.dot(np.linalg.inv(proj_inner_block +
                       np.eye(block.shape[1]) * param), proj_block_ldrs)
            )  # n by r
            level0_preds[:, :, i] = preds.T

        return level0_preds

    def level1_ridge(self, level0_preds_reader, chr_idxs):
        """
        Compute level 1 ridge predictions.

        Parameters:
        ------------
        level0_preds_reader: a h5 data reader and each time read a (n by n_params by n_blocks) array into memory
        chr_idxs: a dictionary of chromosome: [blocks idxs]

        Returns:
        ---------
        chr_preds: a (r by n by #chr) array of predictions

        """
        best_params = np.zeros(self.r)
        chr_preds = np.zeros((self.r, self.n, len(chr_idxs)))
        for j in range(self.r):
            best_params[j] = self._level1_ridge_ldr(
                level0_preds_reader[j], self.resid_ldrs[:, j])
            chr_preds[j] = self._chr_preds_ldr(
                best_params[j], level0_preds_reader[j], self.resid_ldrs[:, j], chr_idxs)

        return chr_preds

    def _level1_ridge_ldr(self, level0_preds, ldr):
        """
        Using cross-validation to select the optimal parameter for each ldr.

        Parameters:
        ------------
        level0_preds: n by n_params by n_blocks matrix for ldr j
        ldr: resid ldr

        Returns:
        ---------
        best_param: the optimal parameter for each ldr

        """
        level0_preds = level0_preds.reshape(level0_preds.shape[0], -1)
        mse = np.zeros((5, len(self.shrinkage)))

        for i, (train_idxs, test_idxs) in enumerate(self.kf.split(level0_preds)):
            train_x = level0_preds[train_idxs]
            test_x = level0_preds[train_idxs]
            train_y = ldr[train_idxs]
            test_y = ldr[test_idxs]
            inner_train_x = np.dot(train_x.T, train_x)
            train_xy = np.dot(train_x.T, train_y)
            for j, param in enumerate(self.shrinkage):
                preditors = np.dot(np.linalg.inv(
                    inner_train_x + np.eye(train_x.shape[1]) * param), train_xy)
                predictions = np.dot(test_x, preditors)
                mse[i, j] = np.mean((test_y - predictions) ** 2)
        mse = np.mean(mse, axis=0)
        min_idx = np.argmin(mse)
        best_param = self.shrinkage[min_idx]

        return best_param

    def _chr_preds_ldr(self, best_param, level0_preds, ldr, chr_idxs):
        """
        Using the optimal parameter to get the chromosome-wise predictions

        Parameters:
        ------------
        best_param: the optimal parameter
        level0_preds: n by n_params by n_blocks matrix for ldr j
        ldr: resid ldr 
        chr_idxs: a dictionary of chromosome: [blocks idxs]

        Returns:
        ---------
        chr_preds_ldr: loco predictions for ldr j

        """
        preds_ldr = np.dot(level0_preds.T, ldr)
        preditors = np.dot(np.linalg.inv(
            level0_preds + np.eye(level0_preds.shape[1]) * best_param), preds_ldr)

        chr_preds_ldr = np.zeros((self.n, len(chr_idxs)))
        for chr, idxs in chr_idxs.items():
            loco_predictors = np.ma.array(preditors, mask=idxs)
            loco_predictions = np.dot(level0_preds, loco_predictors)
            chr_preds_ldr[:, chr-1] = loco_predictions

        chr_preds_ldr = np.sum(chr_preds_ldr, axis=1) - chr_preds_ldr

        return chr_preds_ldr


def merge_blocks(block_sizes, bim):
    """
    Merge adjacent blocks such that we have ~200 blocks with similar size

    Parameters:
    ------------
    block_sizes: a list of LD block sizes 
    bim: a snpinfo df including block idxs

    Returns:
    ---------
    merged_blocks: a list of merged blocks. Each element is the size of merged blocks
    chr_idxs: a dict of chr: block idxs

    """
    chr_idxs_df = bim[['CHR', 'block_idx']].drop_duplicates(
        inplace=True).set_index('block_idx')
    n_blocks = len(block_sizes)
    mean_size = sum(block_sizes) / 200
    merged_blocks = []
    cur_size = 0
    cur_group = []
    last_chr = chr_idxs_df.loc[0, 'CHR']
    for i, block_size in enumerate(block_sizes):
        cur_chr = chr_idxs_df.loc[i, 'CHR']
        if last_chr != cur_chr:
            merged_blocks.append(tuple(cur_group))
            cur_group = [i]
            cur_size = block_size
            last_chr = cur_chr
            continue
        if i < n_blocks - 1:
            if cur_size + block_size <= mean_size or cur_size + block_size // 2 <= mean_size:
                cur_group.append(i)
                cur_size += block_size
            else:
                merged_blocks.append(tuple(cur_group))
                cur_group = [i]
                cur_size = block_size
        else:
            if cur_size + block_size <= mean_size or cur_size + block_size // 2 <= mean_size:
                cur_group.append(i)
                merged_blocks.append(tuple(cur_group))
            else:
                merged_blocks.append(tuple([i]))
        last_chr = cur_chr

    merged_block_sizes = []
    chr_idxs = defaultdict(list)

    for merged_block in merge_blocks:
        block_size = 0
        for idx in merged_block:
            block_size += block_sizes[idx]
            chr_idxs[chr_idxs_df.loc[idx, 'CHR']].append(idx)
        merged_block_sizes.append(block_size)

    return merged_block_sizes, chr_idxs


def check_input(args):
    # required arguments
    if args.bfile is None:
        raise ValueError('--bfile is required.')
    if args.covar is None:
        raise ValueError('--covar is required.')
    if args.ldrs in None:
        raise ValueError('--ldrs is required.')
    if args.partition is None:
        raise ValueError('--partition is required.')
    if args.maf_min is not None:
        if args.maf_min >= 1 or args.maf_min <= 0:
            raise ValueError('--maf must be greater than 0 and less than 1.')
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0.')

    # required files must exist
    if not os.path.exists(args.covar):
        raise FileNotFoundError(f"{args.covar} does not exist.")
    if not os.path.exists(args.ldrs):
        raise FileNotFoundError(f"{args.ldrs} does not exist.")
    if not os.path.exists(args.partition):
        raise FileNotFoundError(f"{args.partition} does not exist.")
    for suffix in ['.bed', '.fam', '.bim']:
        if not os.path.exists(args.bfile + suffix):
            raise FileNotFoundError(f'{args.bfile + suffix} does not exist.')


def run(args, log):
    log.info(f"Read covariates from {args.covar}")
    covar = ds.Covar(args.covar, args.cat_covar_list)
    log.info(f"Read LDRs from {args.ldrs}")
    ldrs = ds.Dataset(args.ldrs)

    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects are in --keep.')
    else:
        keep_idvs = None
    common_idxs = ds.get_common_idxs([covar.index, ldrs.index, keep_idvs])

    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{keep_snps} SNPs are in --extract.")
    else:
        keep_snps = None

    log.info(
        f"Read bfile from {args.bfile} with selected SNPs and individuals.")
    bim, fam, snp_getter = gt.read_plink(
        args.bfile, common_idxs, keep_snps, args.maf_min)
    covar.keep(fam[['FID', 'IID']])  # make sure subjects aligned
    ldrs.keep(fam[['FID', 'IID']])
    log.info(f'{len(covar.data)} subjects are common in these files.\n')

    # genome partition to get LD blocks
    log.info(f"Read genome partition from {args.partition}")
    genome_part = ds.read_geno_part(args.partition)
    log.info(f"{genome_part.shape[0]} genome blocks to partition.")
    num_snps_list, bim = partition_genome(bim, genome_part, log)

    # merge small blocks to get ~200 blocks
    log.info(f"Merge small blocks to get ~200 blocks.")
    num_snps_list, chr_idxs = merge_blocks(num_snps_list, bim)

    # initialize a remover and do level 0 ridge prediction
    relatedness_remover = Relatedness(
        bim.shape[1], np.array(ldrs.data), np.array(covar.data))
    with h5py.File(f'{args.out}_l0_pred_temp.h5', 'w') as file:
        log.info(f'Doing level0 ridge regression ...')
        dset = file.create_dataset('level0_preds',
                                   (ldrs.shape[1],
                                    ldrs.shape[0],
                                    len(relatedness_remover.shrinkage),
                                    len(num_snps_list)),
                                   dtype='float32')
        for i in range(num_snps_list):
            n_snps = num_snps_list[i]
            block_level0_preds = relatedness_remover.level0_ridge_block(
                snp_getter(n_snps))
            dset[:, :, :, i] = block_level0_preds
    log.info(
        f'Save level0 ridge predictions to a temporary file {args.out}_l0_pred_temp.h5')

    # load level 0 predictions by each ldr and do level 1 ridge prediction
    with h5py.File(f'{args.out}_l0_pred_temp.h5', 'r') as file:
        log.info(f'Doing level1 ridge regression ...')
        level0_preds = file['level0_preds']
        chr_preds = relatedness_remover.level1_ridge(level0_preds, chr_idxs)

    with h5py.File(f'{args.out}_ldr_loco_preds.h5', 'w') as file:
        dset = file.create_dataset('ldr_loco_preds', data=chr_preds)
    log.info(
        f'Save level1 loco ridge predictions to {args.out}_ldr_loco_preds.h5')
