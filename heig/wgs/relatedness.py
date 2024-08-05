import os
import h5py
import shutil
import numpy as np
import hail as hl
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold
import heig.input.dataset as ds
from heig.wgs.utils import GProcessor
from hail.linalg import BlockMatrix


"""
TODO: 
1. support parallel
2. spark configure

"""


class Relatedness:
    """
    Remove genetic relatedness by ridge regression.
    Level 0 ridge:
    1. Read SNPs by block
    2. Generate a range of shrinkage parameters
    For each LDR, split the data into 5 folds:
    3. Compute predictors for each pair of LD block and shrinkage parameter
    4. Save the predictors

    Level 1 ridge:
    For each LDR:
    1. Generate a range of shrinkage parameters
    2. Cross-validation to select the optimal shrinkage parameter
    2. Compute predictors using the optimal parameter
    3. Save the LOCO predictors for each chromosome and LDR

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
        self.n, self.r = ldrs.shape
        self.n_blocks = n_blocks
        self.n_params = len(shrinkage)

        shrinkage = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        shrinkage_ = (1 - shrinkage) / shrinkage
        self.shrinkage_level0 = n_snps * shrinkage_
        self.shrinkage_level1 = self.n_params * n_blocks * shrinkage_
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

        for _, test_idxs in self.kf.split(range(self.n)):
            proj_inner_block_ = proj_inner_block - np.dot(block[test_idxs].T, resid_block[test_idxs])
            proj_block_ldrs_ = proj_block_ldrs - np.dot(resid_block[test_idxs].T, self.resid_ldrs[test_idxs])
            for i, param in enumerate(self.shrinkage_level0):
                preds = np.dot(
                    resid_block[test_idxs],
                    np.dot(np.linalg.inv(proj_inner_block_ + np.eye(proj_inner_block_.shape[1]) * param), 
                        proj_block_ldrs_)
                ) # (I-M)Z (Z'(I-M)Z+\lambdaI)^{-1} Z'(I-M)\Xi, (n, r)
                level0_preds[:, test_idxs, i] = preds.T

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
        
        ## get column idxs for each CHR after reshaping
        reshaped_idxs = self._get_reshaped_idxs(chr_idxs)

        ## this can do in parallel
        for j in tqdm(range(self.r), desc=f'{self.r} LDRs'):
            best_params[j] = self._level1_ridge_ldr(level0_preds_reader[j], self.resid_ldrs[:, j])
            chr_preds[j] = self._chr_preds_ldr(
                best_params[j], 
                level0_preds_reader[j], 
                self.resid_ldrs[:, j], 
                reshaped_idxs
            )

        return chr_preds

    def _get_reshaped_idxs(self, chr_idxs):
        """
        Getting predictors for each CHR in reshaped level0 ridge predictions

        Parameters:
        ------------
        chr_idxs: a dictionary of chromosome: [block idxs]

        Returns:
        ---------
        reshaped_idxs: a dictionary of chromosome: [predictor idxs]
        
        """
        reshaped_idxs = dict()
        for chr, idxs in chr_idxs.items():
            chr = int(chr)
            reshaped_idxs_chr = list()
            for idx in idxs:
                reshaped_idxs_chr.extend(range(idx, self.n_params * self.n_blocks, self.n_blocks))
            reshaped_idxs[chr] = reshaped_idxs_chr
        
        return reshaped_idxs
    
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

        ## overall results
        inner_level0_preds = np.dot(level0_preds.T, level0_preds)
        level0_preds_ldr = np.dot(level0_preds.T, ldr)
        
        ## cross validation
        for i, (_, test_idxs) in enumerate(self.kf.split(range(self.n))):
            test_x = level0_preds[test_idxs]
            test_y = ldr[test_idxs]
            inner_train_x = inner_level0_preds - np.dot(test_x.T, test_x)
            train_xy = level0_preds_ldr - np.dot(test_x.T, test_y)
            for j, param in enumerate(self.shrinkage_level1):
                preditors = np.dot(np.linalg.inv(inner_train_x + np.eye(inner_train_x.shape[1]) * param), 
                                   train_xy)
                predictions = np.dot(test_x, preditors)
                mse[i, j] = np.sum((test_y - predictions) ** 2) # squared L2 norm
        mse = np.sum(mse, axis=0) / self.n
        min_idx = np.argmin(mse)
        best_param = self.shrinkage_level1[min_idx]

        return best_param

    def _chr_preds_ldr(self, best_param, level0_preds, ldr, reshaped_idxs):
        """
        Using the optimal parameter to get the chromosome-wise predictions

        Parameters:
        ------------
        best_param: the optimal parameter
        level0_preds: n by n_params by n_blocks matrix for ldr j
        ldr: resid ldr 
        reshaped_idxs: a dictionary of chromosome: [idxs in reshaped predictors]

        Returns:
        ---------
        loco_predictions: loco predictions for ldr j, (n, 22)

        """
        loco_predictions = np.zeros((self.n, 22))
        level0_preds = level0_preds.reshape(self.n, -1)
        level0_preds = level0_preds / np.std(level0_preds, axis=0)

        ## overall results
        inner_level0_preds = np.dot(level0_preds.T, level0_preds)
        preds_ldr = np.dot(level0_preds.T, ldr)

        ## exclude predictors from each CHR
        for chr, idxs in reshaped_idxs.items():
            mask = np.ones(self.n_params * self.n_blocks, dtype=bool)
            mask[idxs] = False
            inner_loco_level0_preds = inner_level0_preds[mask, :][:, mask]
            loco_preds_ldr = preds_ldr[mask]
            loco_preditors = np.dot(
                np.linalg.inv(
                inner_loco_level0_preds + np.eye(inner_loco_level0_preds.shape[1]) * best_param), 
                loco_preds_ldr
                )
            loco_prediction = np.dot(level0_preds[:, mask], loco_preditors)
            loco_predictions[:, chr-1] = loco_prediction

        return loco_predictions


class GenoBlocks:
    """
    Splitting the genome into blocks based on
    1. Pre-defined LD blocks, or
    2. equal-size blocks (REGENIE)
    
    """
    def __init__(self, snps_mt, partition=None, block_size=1000):
        """
        Parameters:
        ------------
        snps_mt: genotype data in MatrixTable
        partition: a pd.DataFrame of genome partition file with columns (without header)
            0: chr, 1: start, 2: end
        block_size: block size (if equal-size block) 
        
        """
        self.snps_mt = snps_mt
        self.partition = partition
        self.block_size = block_size

        if self.partition is not None:
            self.blocks, self.chr_idxs = self._split_ld_blocks()
        else:
            self.blocks, self.chr_idxs = self._split_equal_blocks()

    def _split_ld_blocks(self):
        """
        Splitting the genotype data into pre-defined LD blocks
        Merging into ~200 blocks

        Returns:
        ---------
        blocks: a list of blocks in MatrixTable
        chr_idxs: a dictionary of chromosome: [block idxs]

        """
        if self.partition is None:
            raise ValueError('input a genome partition file by --partition')
        
        n_unique_chrs = len(set(self.partition[0]))
        if n_unique_chrs > 22:
            raise ValueError('sex chromosomes are not supported')
        if n_unique_chrs < 22:
            raise ValueError('genotype data including all autosomes is required')

        if self.partition.shape[0] > 200:
            merged_partition = self._merge_ld_blocks()
        else:
            merged_partition = self.partition
        blocks = []
        chr_idxs = defaultdict(list)
        overall_block_idx = 0
        for _, block in merged_partition.iterrows():
            contig = str(block[0])
            if hl.default_reference == 'GRCh38':
                contig = 'chr' + contig
            start = block[1]
            end = block[2]
            block_mt = self.snps_mt.filter_rows(
                (self.snps_mt.locus.contig == contig) & 
                (self.snps_mt.locus.position >= start) & 
                (self.snps_mt.locus.position < end)
            )
            blocks.append(block_mt)
            chr_idxs[block[0]].append(overall_block_idx)
            overall_block_idx += 1

        return blocks, chr_idxs

    def _merge_ld_blocks(self):
        """
        Merging small LD blocks to ~200 blocks

        Returns:
        ---------
        merged_partition: a pd.DataFrame of merged partition
        
        """
        ## merge blocks by CHR
        n_blocks = self.partition.shape[0]
        n_to_merge = n_blocks // 200
        idx_to_extract = []
        for _, chr_blocks in self.partition.groupby(0):
            n_chr_blocks = chr_blocks.shape[0]
            idx = chr_blocks.index
            idx_to_extract.append(idx[0])
            if n_chr_blocks >= n_to_merge:
                idx_to_extract += list(idx[n_to_merge: : n_to_merge])
            if idx[-1] not in idx_to_extract:
                idx_to_extract.append(idx[-1])
        merged_partition = self.partition.loc[idx_to_extract].copy()

        ## the end of the current block should be the start of the next
        updated_end = list()
        for _, chr_blocks in merged_partition.groupby(0):
            updated_end.extend(list(chr_blocks.iloc[1:, 1]))
            updated_end.append(chr_blocks.iloc[-1, 2])
        merged_partition[2] = updated_end

        return merged_partition

    def _split_equal_blocks(self):
        """
        Splitting the genotype data into approximately equal-size blocks

        Returns:
        ---------
        blocks: a list of blocks in MatrixTable
        chr_idxs: a dictionary of chromosome: [block idxs]
        
        """
        blocks = []
        chr_idxs = defaultdict(list)
        overall_block_idx = 0
        chrs = set(self.snps_mt.aggregate_rows(hl.agg.collect(self.snps_mt.locus.contig))) # slow
        
        if len(chrs) > 22:
            raise ValueError('sex chromosomes are not supported')
        if len(chrs) < 22:
            raise ValueError('genotype data including all autosomes is required')

        for chr in chrs:
            snps_mt_chr = self.snps_mt.filter_rows(self.snps_mt.locus.contig == chr)
            snps_mt_chr = snps_mt_chr.add_row_index()
            n_variants = snps_mt_chr.count_rows()
            n_blocks = (n_variants // self.block_size) + int(n_variants % self.block_size > 0)
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


def check_input(args, log):
    # required arguments
    if args.bfile is None and args.geno_mt is None:
        raise ValueError('--bfile or --geno-mt is required.')
    if args.covar is None:
        raise ValueError('--covar is required.')
    if args.ldrs is None:
        raise ValueError('--ldrs is required.')
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
    if args.partition is not None and not os.path.exists(args.partition):
        raise FileNotFoundError(f"{args.partition} does not exist.")
    if args.geno_mt is not None and not os.path.exists(args.geno_mt):
            raise FileNotFoundError(f"{args.geno_mt} does not exist.")
    if args.bfile is not None:
        for suffix in ['.bed', '.fam', '.bim']:
            if not os.path.exists(args.bfile + suffix):
                raise FileNotFoundError(f'{args.bfile + suffix} does not exist.')
            
    if args.bsize is not None and args.bsize < 1000:
        raise ValueError('--bsize should be no less than 1000.')
    elif args.bsize is None:
        args.bsize = 1000
    
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

        # split the genome into blocks
        if args.partition is not None:
            genome_part = ds.read_geno_part(args.partition)
            log.info(f"{genome_part.shape[0]} genome blocks to partition ...")
            if genome_part.shape[0] > 200:
                log.info(f'Merge into ~200 blocks.')
        else:
            genome_part = None
            log.info(f"Partition the genome into blocks of size ~{args.bsize} ...")
        
        geno_block = GenoBlocks(gprocessor.snps_mt, genome_part, args.bsize)
        blocks, chr_idxs = geno_block.blocks, geno_block.chr_idxs

        # initialize a remover and do level 0 ridge prediction
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
                block = BlockMatrix.from_entry_expr(block.GT.n_alt_alleles(), mean_impute=True) # (m, n)
                block = block.to_numpy().T
                block_level0_preds = relatedness_remover.level0_ridge_block(block)
                dset[:, :, :, i] = block_level0_preds
        log.info(f'Save level0 ridge predictions to a temporary file {args.out}_l0_pred_temp.h5')

        # load level 0 predictions by each ldr and do level 1 ridge prediction
        with h5py.File(f'{args.out}_l0_pred_temp.h5', 'r') as file:
            log.info(f'Doing level1 ridge regression ...')
            level0_preds_reader = file['level0_preds']
            chr_preds = relatedness_remover.level1_ridge(level0_preds_reader, chr_idxs)

        with h5py.File(f'{args.out}_ldr_loco_preds.h5', 'w') as file:
            file.create_dataset('ldr_loco_preds', data=chr_preds, dtype='float32')
        log.info(f'Save level1 loco ridge predictions to {args.out}_ldr_loco_preds.h5')
    finally:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            log.info(f'Removed preprocessed genotype data at {temp_path}')