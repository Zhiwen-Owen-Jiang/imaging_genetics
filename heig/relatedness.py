import os
import h5py
import shutil
import numpy as np
import pandas as pd
import hail as hl
from tqdm import tqdm
from math import ceil
from collections import defaultdict
from sklearn.model_selection import KFold
import heig.input.dataset as ds
from heig.ldmatrix import partition_genome
from heig.wgs.utils import GProcessor
from hail.linalg import BlockMatrix


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

    def __init__(self, n_snps, n_blocks, ldrs, covar):
        """
        num_snps_part: a dict of chromosome: [LD block sizes]
        snp_getter: a generator for getting SNPs
        n_blocks: a positive number of genotype blocks
        n_snps: a positive number of total array snps
        ldrs: n by r matrix
        covar: n by p matrix (preprocessed, including the intercept)

        """
        ## these may come from the null model
        # shrinkage = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
        shrinkage = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        shrinkage_ = (1 - shrinkage) / shrinkage
        self.shrinkage_level0 = n_snps * shrinkage_

        # shrinkage = np.array([0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99])
        # shrinkage_ = (1 - shrinkage) / shrinkage
        self.shrinkage_level1 = len(shrinkage) * n_blocks * shrinkage_
        # self.shrinkage_level1 = shrinkage_
        self.kf = KFold(n_splits=5)
        self.n, self.r = ldrs.shape

        self.inner_covar_inv = np.linalg.inv(np.dot(covar.T, covar))
        self.resid_ldrs = ldrs - np.dot(np.dot(covar, self.inner_covar_inv), 
                                        np.dot(covar.T, ldrs))
        self.covar = covar

    def level0_ridge_block(self, block):
        """
        Compute level 0 ridge prediction for a genotype block.
        Missing values in each block have been imputed.

        Parameters:
        ------------
        block: m by n matrix of a genotype block

        Returns:
        ---------
        level0_preds: a (r by n by 5) array of predictions

        """
        level0_preds = np.zeros((self.r, self.n, len(self.shrinkage_level0)))
        block_covar = np.dot(block.T, self.covar) # Z'X, (m, p)
        resid_block = block - np.dot(np.dot(self.covar, self.inner_covar_inv), 
                                     block_covar.T) # (I-M)Z = Z-X(X'X)^{-1}X'Z, (n, m)
        resid_block = resid_block / np.std(resid_block, axis=0)

        proj_inner_block = np.dot(block.T, resid_block) # Z'(I-M)Z, (m, m)
        proj_block_ldrs = np.dot(resid_block.T, self.resid_ldrs) # Z'(I-M)\Xi, (m, r)
        for i, param in enumerate(self.shrinkage_level0):
            preds = np.dot(
                resid_block,
                np.dot(np.linalg.inv(proj_inner_block + np.eye(block.shape[1]) * param), 
                       proj_block_ldrs)
            ) # (I-M)Z (Z'(I-M)Z+\lambdaI)^{-1} Z'(I-M)\Xi, (n, r)
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
            best_params[j] = self._level1_ridge_ldr(level0_preds_reader[j], self.resid_ldrs[:, j])
            chr_preds[j] = self._chr_preds_ldr(
                best_params[j], 
                level0_preds_reader[j], 
                self.resid_ldrs[:, j], 
                chr_idxs
            )

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
        level0_preds = level0_preds / np.std(level0_preds, axis=0)
        mse = np.zeros((5, len(self.shrinkage_level1)))

        for i, (train_idxs, test_idxs) in enumerate(self.kf.split(level0_preds)):
            train_x = level0_preds[train_idxs]
            test_x = level0_preds[test_idxs]
            train_y = ldr[train_idxs]
            test_y = ldr[test_idxs]
            inner_train_x = np.dot(train_x.T, train_x)
            train_xy = np.dot(train_x.T, train_y)
            for j, param in enumerate(self.shrinkage_level1):
                preditors = np.dot(np.linalg.inv(inner_train_x + np.eye(train_x.shape[1]) * param), 
                                   train_xy)
                predictions = np.dot(test_x, preditors)
                mse[i, j] = np.mean((test_y - predictions) ** 2)
        mse = np.mean(mse, axis=0)
        min_idx = np.argmin(mse)
        best_param = self.shrinkage_level1[min_idx]

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
        chr_predictions: loco predictions for ldr j

        """
        chr_predictions = np.zeros((self.n, len(chr_idxs)))
        for chr, idxs in chr_idxs.items():
            chr = int(chr)
            chr_level0_preds = level0_preds[:, :, idxs]
            chr_level0_preds = chr_level0_preds.reshape(chr_level0_preds.shape[0], -1)
            chr_level0_preds = chr_level0_preds / np.std(chr_level0_preds, axis=0)
            inner_chr_level0_preds = np.dot(chr_level0_preds.T, chr_level0_preds)
            chr_preds_ldr = np.dot(chr_level0_preds.T, ldr)
            chr_preditors = np.dot(
                np.linalg.inv(
                inner_chr_level0_preds + np.eye(chr_level0_preds.shape[1]) * best_param), 
                chr_preds_ldr
                )
            chr_prediction = np.dot(chr_level0_preds, chr_preditors)
            chr_predictions[:, chr-1] = chr_prediction
        chr_predictions = np.sum(chr_predictions, axis=1) - chr_predictions

        return chr_predictions


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
    chr_idxs_df = bim[['CHR', 'block_idx']].drop_duplicates(inplace=True).set_index('block_idx')
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


def split_blocks(snps_mt, block_size=1000):
    """
    Splitting the genotype data into equal-size blocks
    
    """
    blocks = []
    chr_idxs = defaultdict(list)
    overall_block_idx = 0
    chrs = set(snps_mt.aggregate_rows(hl.agg.collect(snps_mt.locus.contig))) # slow
    
    if len(chrs) > 22:
        raise ValueError('sex chromosomes are not supported')
    if len(chrs) < 22:
        raise ValueError('genotype data including all autosomes is required')

    for chr in chrs:
        snps_mt_chr = snps_mt.filter_rows(snps_mt.locus.contig == chr)
        snps_mt_chr = snps_mt_chr.add_row_index()
        n_variants = snps_mt_chr.count_rows()
        n_blocks = (n_variants // block_size) + int(n_variants % block_size > 0)
        block_size_chr = n_variants // n_blocks

        for block_idx in range(n_blocks):
            start = block_idx * block_size_chr
            if block_idx == n_blocks - 1:
                end = n_variants
            else:
                end = (block_idx + 1) * block_size_chr
            block_mt = snps_mt_chr.filter_rows(
                (snps_mt_chr.row_idx >= start) & (snps_mt_chr.row_idx < end)
            )
            blocks.append(block_mt)
            chr_idxs[chr].append(overall_block_idx)
            overall_block_idx += 1

    return blocks, chr_idxs


def get_ld_blocks(snps_mt, start, end, bim):
    bim = bim.iloc[start: end, :]
    if len(set(bim[0])) != 1:
        raise ValueError('the LD block is cross chromosomes')
    chr = bim[0]
    start = bim.iloc[0, 3]
    end = bim.iloc[-1, 3]
    snps_mt_block = snps_mt.filter_rows((snps_mt.locus.contig == chr) & 
                                        (snps_mt.locus.position >= start) & 
                                        (snps_mt.locus.position < end))
    ld_block = BlockMatrix.from_entry_expr(snps_mt_block.flipped_n_alt_alleles, mean_impute=True) # (m, n)
    ld_block = ld_block.to_numpy().T
    
    return ld_block


def check_input(args, log):
    # required arguments
    if args.bfile is None and args.geno_mt is None:
        raise ValueError('--bfile or --geno-mt is required.')
    if args.covar is None:
        raise ValueError('--covar is required.')
    if args.ldrs is None:
        raise ValueError('--ldrs is required.')
    if args.partition is None:
        raise ValueError('--partition is required.')
    if args.maf_min is not None:
        if args.maf_min > 0.5 or args.maf_min < 0:
            raise ValueError('--maf-min must be greater than 0 and less than 0.5')
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0.')

    # required files must exist
    if not os.path.exists(args.covar):
        raise FileNotFoundError(f"{args.covar} does not exist.")
    if not os.path.exists(args.ldrs):
        raise FileNotFoundError(f"{args.ldrs} does not exist.")
    if not os.path.exists(args.partition):
        raise FileNotFoundError(f"{args.partition} does not exist.")
    if args.geno_mt is not None:
        if not os.path.exists(args.geno_mt):
            raise FileNotFoundError(f"{args.geno_mt} does not exist.")
    if args.bfile is not None:
        for suffix in ['.bed', '.fam', '.bim']:
            if not os.path.exists(args.bfile + suffix):
                raise FileNotFoundError(f'{args.bfile + suffix} does not exist.')
    
    temp_path = 'temp'
    i = 0
    while os.path.exists(temp_path + str(i)):
        i += 1
    temp_path += str(i)
        
    if args.grch37 is None or not args.grch37:
        geno_ref = 'GRCh38'
    else:
        geno_ref = 'GRCh37'
    log.info(f'Set {geno_ref} as the reference genome.')

    return temp_path, geno_ref


def run(args, log):
    # check input and configure hail
    temp_path, geno_ref = check_input(args, log)

    # read LDRs and covariates
    log.info(f'Read LDRs from {args.ldrs}')
    ldrs = ds.Dataset(args.ldrs)
    log.info(f'{ldrs.data.shape[1]} LDRs and {ldrs.data.shape[0]} subjects.')
    if args.n_ldrs is not None:
        ldrs.data = ldrs.data.iloc[:, :args.n_ldrs]
        if ldrs.data.shape[1] > args.n_ldrs:
            log.info(f'WARNING: --n-ldrs greater than #LDRs, use all LDRs.')
        else:
            log.info(f'Keep the top {args.n_ldrs} LDRs.')        

    log.info(f'Read covariates from {args.covar}')
    covar = ds.Covar(args.covar, args.cat_covar_list)

    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None
    common_ids = ds.get_common_idxs(ldrs.data.index, covar.data.index, keep_idvs, single_id=True)

    # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} SNPs in --extract.")
    else:
        keep_snps = None

    # read genotype data
    spark_conf = {
    'spark.executor.memory': '8g',
    'spark.driver.memory': '8g',
    'spark.master': 'local[8]'
    }
    hl.init(quiet=True, spark_conf=spark_conf)
    hl.default_reference = geno_ref

    if args.bfile is not None:
        log.info(f'Read bfile from {args.bfile}')
        gprocessor = GProcessor.import_plink(args.bfile, geno_ref, 
                                            maf_min=args.maf_min)
    elif args.geno_mt is not None:
        log.info(f'Read genotype data from {args.geno_mt}')
        gprocessor = GProcessor.read_matrix_table(args.geno_mt, geno_ref,
                                                  maf_min=args.maf_min)
    log.info(f"Processing genetic data ...")
    gprocessor.extract_snps(keep_snps)
    gprocessor.extract_idvs(common_ids)
    gprocessor.do_processing(mode='gwas')

    if not args.not_save_genotype_data:
        log.info(f'Save preprocessed genotype data to {temp_path}')
        gprocessor.save_interim_data(temp_path)
    
    try:
        gprocessor.check_valid()
        snps_mt_ids = gprocessor.subject_id()
        ldrs.to_single_index()
        covar.to_single_index()
        ldrs.keep(snps_mt_ids)
        covar.keep(snps_mt_ids)
        covar.cat_covar_intercept()
        log.info(f'{len(snps_mt_ids)} common subjects in the data.')
        log.info(f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept).")

        # bim = gprocessor.get_bim()
        # # genome partition to get LD blocks
        # log.info(f"Read genome partition from {args.partition}")
        # genome_part = ds.read_geno_part(args.partition)
        # log.info(f"{genome_part.shape[0]} genome blocks to partition.")
        # # bim = pd.read_csv(args.bfile + '.bim', sep='\s+', header=None)
        # num_snps_list, bim = partition_genome(bim, genome_part, log)

        # # merge small blocks to get ~200 blocks
        # log.info(f"Merge small blocks to get ~200 blocks.")
        # num_snps_list, chr_idxs = merge_blocks(num_snps_list, bim)

        # initialize a remover and do level 0 ridge prediction
        blocks, chr_idxs = split_blocks(gprocessor.snps_mt)
        n_variants = gprocessor.snps_mt.count_rows()
        n_blocks = len(blocks)
        relatedness_remover = Relatedness(
            n_variants, n_blocks, np.array(ldrs.data), np.array(covar.data)
        )
        with h5py.File(f'{args.out}_l0_pred_temp.h5', 'w') as file:
            log.info(f'Doing level0 ridge regression ...')
            dset = file.create_dataset('level0_preds',
                                        (ldrs.data.shape[1],
                                        ldrs.data.shape[0],
                                        len(relatedness_remover.shrinkage_level0),
                                        n_blocks),
                                    dtype='float32')
            for i, block in enumerate(tqdm(blocks, desc=f'{n_blocks} blocks')):
                # block = get_ld_blocks(gprocessor.snps_mt, start, end, bim)
                block = BlockMatrix.from_entry_expr(block.GT.n_alt_alleles(), mean_impute=True) # (m, n)
                block = block.to_numpy().T
                block_level0_preds = relatedness_remover.level0_ridge_block(block)
                dset[:, :, :, i] = block_level0_preds
        log.info(f'Save level0 ridge predictions to a temporary file {args.out}_l0_pred_temp.h5')

        # load level 0 predictions by each ldr and do level 1 ridge prediction
        with h5py.File(f'{args.out}_l0_pred_temp.h5', 'r') as file:
            log.info(f'Doing level1 ridge regression ...')
            level0_preds = file['level0_preds']
            chr_preds = relatedness_remover.level1_ridge(level0_preds, chr_idxs)

        with h5py.File(f'{args.out}_ldr_loco_preds.h5', 'w') as file:
            dset = file.create_dataset('ldr_loco_preds', data=chr_preds)
        log.info(f'Save level1 loco ridge predictions to {args.out}_ldr_loco_preds.h5')
    finally:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            log.info(f'Removed preprocessed genotype data at {temp_path}')
