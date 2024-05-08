import os
import pickle
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import heig.input.genotype as gt
import heig.input.dataset as ds


def parse_ld_input(arg):
    """
    Parsing the LD matrix files. 

    Parameters:
    ------------
    arg: the prefix with indices, seperated by a comma, 
    e.g., `ldmatrix/ukb_white_exclude_phase123_25k_sub_chr{1:22}_LD1`

    Returns:
    ---------
    A list of parsed gwas files 

    """
    p1 = r'{(.*?)}'
    p2 = r'({.*})'
    match = re.search(p1, arg)
    if match:
        file_range = match.group(1)
    else:
        raise ValueError(('if multiple LD matrices are provided, '
                          '--ld or --ld-inv should be specified using `{}`, '
                          'e.g. `prefix_chr{stard:end}`'))
    start, end = [int(x) for x in file_range.split(":")]
    ld_files = [re.sub(p2, str(i), arg) for i in range(start, end + 1)]

    return ld_files


class LDmatrix:
    def __init__(self, ld_prefix):
        """
        Loading an existing LD matrix

        Parameters:
        ------------
        ld_prefix: prefix of LD matrix file 

        """
        if '{' in ld_prefix and '}' in ld_prefix:
            ld_prefix_list = parse_ld_input(ld_prefix)
        else:
            ld_prefix_list = [ld_prefix]
        self.ldinfo = self._merge_ldinfo(ld_prefix_list)
        self.data = self._read_as_generator(ld_prefix_list)
        self.block_sizes, self.block_ranges = self._get_block_info(self.ldinfo)

    def _read_ldinfo(self, prefix):
        ldinfo = pd.read_csv(f"{prefix}.ldinfo", delim_whitespace=True, header=None,
                             names=['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2', 'MAF',
                                    'block_idx', 'block_idx2', 'ldscore'])
        if not ldinfo.groupby('CHR')['POS'].apply(lambda x: x.is_monotonic_increasing).all():
            raise ValueError(
                f'the SNPs in each chromosome are not sorted or there are duplicated SNPs')

        return ldinfo

    def _merge_ldinfo(self, prefix_list):
        """
        Merging multiple LD matrices with the current one

        Parameters:
        ------------
        prefix_list: a list of prefix of ld file

        """
        if len(prefix_list) == 0:
            raise ValueError('nothing in the LD list')

        ldinfo = self._read_ldinfo(prefix_list[0])
        for prefix in prefix_list[1:]:
            ldinfo_i = self._read_ldinfo(prefix)
            ldinfo_i['block_idx'] += ldinfo.loc[ldinfo.index[-1], 'block_idx'] + 1
            ldinfo = pd.concat([ldinfo, ldinfo_i], axis=0, ignore_index=True)
        return ldinfo

    def _read_as_generator(self, prefix_list):
        for prefix in prefix_list:
            file_path = f"{prefix}.ldmatrix"
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                for item in data:
                    yield item

    @classmethod
    def _get_block_info(cls, ldinfo):
        block_sizes = pd.value_counts(
            ldinfo['block_idx']).sort_index().to_list()
        block_ranges = []
        begin, end = 0, 0
        for block_size in block_sizes:
            begin = end
            end += block_size
            block_ranges.append((begin, end))
        return block_sizes, block_ranges

    def extract(self, snps):
        """
        Extracting SNPs from the LD matrix

        Parameters:
        ------------
        snps: a list/set of rdID 

        Returns:
        ---------
        Updated LD matrix and LD info

        """
        self.ldinfo = self.ldinfo.loc[self.ldinfo['SNP'].isin(snps)]
        block_dict = {k: g["block_idx2"].tolist()
                      for k, g in self.ldinfo.groupby("block_idx")}
        self.block_sizes, self.block_ranges = self._get_block_info(self.ldinfo)
        self.data = (block[block_dict[i]]
                     for i, block in enumerate(self.data) if i in block_dict)

    def merge_blocks(self):
        """
        Merge small blocks such that we have ~200 blocks with similar size

        Parameters:
        ------------
        block_ranges: a dictionary of (begin, end) of each block

        Returns:
        ---------
        merged_blocks: a list of merged blocks

        """
        n_blocks = len(self.block_sizes)
        mean_size = sum(self.block_sizes) / 200
        merged_blocks = []
        cur_size = 0
        cur_group = []
        for i, block_size in enumerate(self.block_sizes):
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

        return merged_blocks


class LDmatrixBED(LDmatrix):
    def __init__(self, num_snps_part, ldinfo, snp_getter, prop, inv=False):
        """
        Making an LD matrix from a bed file

        Parameters:
        ------------
        num_snps_part: a list of number of SNPs in each LD block
        ldinfo: SNP information
        snp_getter: a generator for getting SNPs
        prop: proportion of variance to keep for each LD block
        inv: if take inverse or not 

        """
        self.data = []
        ldscore = []
        for _, num in enumerate(tqdm(num_snps_part, desc=f'Making {len(num_snps_part)} LD blocks')):
            block = snp_getter(num)
            block = self._fill_na(block)
            corr = np.atleast_2d(np.corrcoef(block.T))
            ldscore_i = self._estimate_ldscore(corr, block.shape[0])
            ldscore.append(ldscore_i)
            values, bases = self._truncate(corr, prop)
            if inv:
                bases = bases * np.sqrt(values ** -1)
            else:
                bases = bases * np.sqrt(values)
            self.data.append(bases)
        ldinfo['ldscore'] = np.concatenate(ldscore, axis=None)
        self.ldinfo = ldinfo

    def _truncate(self, block, prop):
        values, bases = np.linalg.eigh(block)
        values = np.flip(values)
        bases = np.flip(bases, axis=1)
        prop_var = np.cumsum(values) / np.sum(values)
        idxs = (prop_var <= prop) & (values > 0)
        n_idxs = sum(idxs) + 1
        values = values[:n_idxs]
        bases = bases[:, :n_idxs]

        return values, bases

    def _fill_na(self, block):
        """
        Filling missing genotypes with the mean

        """
        block_avg = np.nanmean(block, axis=0)
        nanidx = np.where(np.isnan(block))
        block[nanidx] = block_avg[nanidx[1]]

        return block

    def _estimate_ldscore(self, corr, n):
        """
        Estimating LD score from the LD matrix
        The Pearson correlation is adjusted by r2 - (1 - r2) / (N - 2)

        """
        raw_ld = np.sum(corr ** 2, axis=0)
        adj_ld = raw_ld - (corr.shape[0] - raw_ld) / (n - 2)

        return adj_ld

    def save(self, out, inv, prop):
        if not inv:
            prefix = f"{out}_ld_regu{int(prop*100)}"
        else:
            prefix = f"{out}_ld_inv_regu{int(prop*100)}"

        with open(f"{prefix}.ldmatrix", 'wb') as file:
            pickle.dump(self.data, file)
        self.ldinfo['MAF'] = self.ldinfo['MAF'].astype(np.float)
        self.ldinfo.to_csv(f"{prefix}.ldinfo", sep='\t', index=None,
                           header=None, float_format='%.4f')

        return prefix


def partition_genome(ld_bim, part, log):
    """
    ld_bim: a pd Dataframe of LD matrix SNP information
    part: a pd Dataframe of LD block annotation
    log: a logger

    """
    num_snps_part = []
    end = -1
    ld_bim['block_idx'] = None
    ld_bim['block_idx2'] = None
    abs_begin = 0
    abs_end = 0
    n_skipped_blocks = 0
    block_idx = 0
    for i in range(part.shape[0]):
        cand = list(ld_bim.loc[ld_bim['CHR'] == part.iloc[i, 0], 'POS'])
        begin = end
        end = find_loc(cand, part.iloc[i, 2])
        if end < begin:
            begin = -1
        if end > begin:
            block_size = end - begin
            if block_size < 2000:
                sub_blocks = [(begin, end)]
            else:
                log.info((f'A large LD block with size {block_size}, '
                          'evenly partition it to small blocks with size ~1000.'))
                sub_blocks = get_sub_blocks(begin, end)
            for sub_block in sub_blocks:
                sub_begin, sub_end = sub_block
                sub_block_size = sub_end - sub_begin
                num_snps_part.append(sub_block_size)
                if not abs_begin and not abs_end:
                    abs_begin = sub_begin + 1
                    abs_end = sub_end + 1
                else:
                    abs_begin = abs_end
                    abs_end += sub_block_size
                ld_bim.loc[ld_bim.index[abs_begin: abs_end],
                           'block_idx'] = block_idx
                ld_bim.loc[ld_bim.index[abs_begin: abs_end],
                           'block_idx2'] = range(sub_block_size)
                block_idx += 1
        else:
            n_skipped_blocks += 1
    log.info(f'{n_skipped_blocks} blocks with no SNP are skipped.')

    return num_snps_part, ld_bim


def find_loc(num_list, target):
    l = 0
    r = len(num_list) - 1
    while l <= r:
        mid = (l + r) // 2
        if num_list[mid] == target:
            return mid
        elif num_list[mid] > target:
            r = mid - 1
        else:
            l = mid + 1
    return r


def get_sub_blocks(begin, end):
    block_size = end - begin
    n_sub_blocks = block_size // 1000
    sub_block_size = block_size // n_sub_blocks
    sub_blocks = []
    for _ in range(n_sub_blocks - 1):
        temp_end = begin + sub_block_size
        sub_blocks.append((begin, temp_end))
        begin = temp_end
    sub_blocks.append((begin, end))

    return sub_blocks


def check_input(args):
    # required arguments
    if args.bfile is None:
        raise ValueError('--bfile is required')
    if args.partition is None:
        raise ValueError('--partition is required')
    if args.ld_regu is None:
        raise ValueError('--ld-regu is required')
    if args.maf_min is not None:
        if args.maf_min >= 1 or args.maf_min <= 0:
            raise ValueError('--maf must be greater than 0 and less than 1')

    # check file/directory exists
    if not os.path.exists(args.partition):
        raise FileNotFoundError(f'{args.partition} does not exist')

    # processing some arguments
    try:
        ld_bfile, ld_inv_bfile = args.bfile.split(',')
    except:
        raise ValueError(
            'two bfiles must be provided with --bfile and separated with a comma')
    for suffix in ['.bed', '.fam', '.bim']:
        if not os.path.exists(ld_bfile + suffix):
            raise FileNotFoundError(f'{ld_bfile + suffix} does not exist')
        if not os.path.exists(ld_inv_bfile + suffix):
            raise FileNotFoundError(f'{ld_inv_bfile + suffix} does not exist')

    try:
        ld_regu, ld_inv_regu = [float(x) for x in args.ld_regu.split(',')]
    except:
        raise ValueError(('two regularization levels must be provided with --prop '
                          'and separated with a comma'))
    if ld_regu >= 1 or ld_regu <= 0 or ld_inv_regu >= 1 or ld_inv_regu <= 0:
        raise ValueError(
            'both regularization levels must be greater than 0 and less than 1')

    return ld_bfile, ld_inv_bfile, ld_regu, ld_inv_regu


def read_process_snps(bim_dir, log):
    log.info(f"Read SNP list from {bim_dir} and remove duplicated SNPs.")
    ld_bim = pd.read_csv(bim_dir, sep='\s+', header=None,
                         names=['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2'],
                         dtype={'A1': 'category', 'A2': 'category'})
    ld_bim.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
    log.info(
        f'{ld_bim.shape[0]} SNPs remaining after removing duplicated SNPs.')

    return ld_bim


def read_process_idvs(fam_dir):
    ld_fam = pd.read_csv(fam_dir, sep='\s+', header=None,
                         names=['FID', 'IID', 'FATHER', 'MOTHER', 'GENDER', 'TRAIT'], 
                         dtype={'FID': str, 'IID': str})
    ld_fam = ld_fam.set_index(['FID', 'IID'])

    return ld_fam


def filter_maf(ld_bfile, ld_keep_snp, ld_keep_idv,
               ld_inv_bfile, ld_inv_keep_snp, ld_inv_keep_idv, min_maf):
    ld_bim2, *_ = gt.read_plink(ld_bfile, ld_keep_snp, ld_keep_idv)
    ld_inv_bim2, *_ = gt.read_plink(ld_inv_bfile, ld_inv_keep_snp, ld_inv_keep_idv)
    common_snps = ld_bim2.loc[(ld_bim2['MAF'] >= min_maf) & (
        ld_inv_bim2['MAF'] >= min_maf)]

    return common_snps


def run(args, log):
    # checking if arguments are valid
    ld_bfile, ld_inv_bfile, ld_regu, ld_inv_regu = check_input(args)

    # reading and removing duplicated SNPs
    ld_bim = read_process_snps(ld_bfile + '.bim', log)
    ld_inv_bim = read_process_snps(ld_inv_bfile + '.bim', log)

    # merging two SNP lists
    ld_merged = ld_bim.merge(ld_inv_bim, on=['SNP', 'A1', 'A2'])
    log.info(
        f"{ld_merged.shape[0]} SNPs are common in two bfiles with identical A1 and A2.")

    # extracting SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        ld_merged = ld_merged.loc[ld_merged['SNP'].isin(keep_snps['SNP'])]
        log.info(f"{ld_merged.shape[0]} SNPs in --extract.")
    ld_keep_snp = ld_bim.merge(ld_merged, on='SNP')
    ld_inv_keep_snp = ld_inv_bim.merge(ld_merged, on='SNP')

    # keeping individuals
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
        ld_fam = read_process_idvs(ld_bfile + '.fam')
        ld_keep_idv = ld_fam.loc[keep_idvs]
        log.info(f'{len(ld_keep_idv)} subjects kept in {ld_bfile}')
        ld_inv_fam = read_process_idvs(ld_inv_bfile + '.fam')
        ld_inv_keep_idv = ld_inv_fam.loc[keep_idvs]
        log.info(f'{len(ld_inv_keep_idv)} subjects kept in {ld_inv_bfile}')
    else:
        ld_keep_idv, ld_inv_keep_idv = None, None

    # filtering rare SNPs
    if args.maf_min is not None:
        log.info(f"Removing SNPs with MAF < {args.maf_min} ...")
        common_snps = filter_maf(ld_bfile, ld_keep_snp, 
                                 ld_keep_idv, ld_inv_bfile, 
                                 ld_inv_keep_snp, ld_inv_keep_idv, args.maf_min)
        log.info(f"{len(common_snps)} SNPs remaining.")

    # reading bfiles
    log.info(f"Read bfile from {ld_bfile} with selected SNPs and individuals.")
    ld_bim, _, ld_snp_getter = gt.read_plink(
        ld_bfile, common_snps, ld_keep_idv)
    log.info(
        f"Read bfile from {ld_inv_bfile} with selected SNPs and individuals.")
    ld_inv_bim, _, ld_inv_snp_getter = gt.read_plink(
        ld_inv_bfile, common_snps, ld_inv_keep_idv)

    # reading and doing genome partition
    log.info(f"\nRead genome partition from {args.partition}")
    genome_part = ds.read_geno_part(args.partition)
    log.info(f"{genome_part.shape[0]} genome blocks to partition.")
    num_snps_part, ld_bim = partition_genome(ld_bim, genome_part, log)
    ld_inv_bim['block_idx'] = ld_bim['block_idx']
    ld_inv_bim['block_idx2'] = ld_bim['block_idx2']
    log.info((f"{sum(num_snps_part)} SNPs partitioned into {len(num_snps_part)} blocks, "
              f"with the biggest one {np.max(num_snps_part)} SNPs."))

    # making LD matrix and its inverse
    log.info(
        f"Regularization {ld_regu} for LD matrix, and {ld_inv_regu} for LD inverse matrix.")
    log.info(f"Making LD matrix and its inverse ...\n")
    ld = LDmatrixBED(num_snps_part, ld_bim, ld_snp_getter, ld_regu)
    ld_inv = LDmatrixBED(num_snps_part, ld_inv_bim,
                         ld_inv_snp_getter, ld_inv_regu, inv=True)

    ld_prefix = ld.save(args.out, False, ld_regu)
    log.info(f"Save LD matrix to {ld_prefix}.ldmatrix")
    log.info(f"Save LD matrix info to {ld_prefix}.ldinfo")

    ld_inv_prefix = ld_inv.save(args.out, True, ld_inv_regu)
    log.info(f"Save LD inverse matrix to {ld_inv_prefix}.ldmatrix")
    log.info(f"Save LD inverse matrix info to {ld_inv_prefix}.ldinfo")
