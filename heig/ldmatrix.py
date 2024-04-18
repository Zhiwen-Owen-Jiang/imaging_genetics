import os
import pickle
import re
import pandas as pd
import numpy as np
from .parse import PlinkBIMFile, PlinkFAMFile, PlinkBEDFile


"""
required arguments:
--partition
--bfile
--prop
--out

"""


"""
TODO: debug

"""


def parse_ld_input(arg):
    """
    Parsing the LD matrix files. 

    Parameters:
    ------------
    arg: the prefix with indices, seperated by a comma, e.g., `ldmatrix/ukb_white_exclude_phase123_25k_sub_chr{1:22}_LD1`

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
        raise ValueError('if multiple LD matrices are provided, --ld or --ld-inv should be specified using `{}`, \
                         e.g. `prefix_chr{stard:end}`')
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
        self.ld_info = self._merge_ldinfo(ld_prefix_list)
        self.data = self._read_as_generator(ld_prefix_list)
        self.block_sizes, self.block_ranges = self._get_block_info(self.ld_info)
        

    def _read_ldinfo(self, prefix):
        ld_info = pd.read_csv(f"{prefix}.ldinfo", delim_whitespace=True, header=None, 
                                   names=['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2', 'MAF',
                                          'block_idx', 'block_idx2', 'ldscore'])
        if not ld_info.groupby('CHR')['POS'].apply(lambda x: x.is_monotonic_increasing).all():
            raise ValueError(f'The SNPs in each chromosome are not sorted or there are duplicated SNPs')
        
        return ld_info
        

    def _merge_ldinfo(self, prefix_list):
        """
        Merging multiple LD matrices with the current one
        
        Parameters:
        ------------
        prefix_list: a list of prefix of ld file
        
        """
        if len(prefix_list) == 0:
            raise ValueError('There is nothing in the ld list')

        ld_info = self._read_ldinfo(prefix_list[0])
        for prefix in prefix_list[1:]:
            ld_info_i = self._read_ldinfo(prefix)
            ld_info_i['block_idx'] += ld_info.loc[ld_info.index[-1], 'block_idx'] + 1
            ld_info = pd.concat([ld_info, ld_info_i], axis=0, ignore_index=True)
        return ld_info
        

    def _read_as_generator(self, prefix_list):
        for prefix in prefix_list:
            file_path = f"{prefix}.ldmatrix"
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                for item in data:
                    yield item


    @classmethod
    def _get_block_info(cls, ld_info):
        block_sizes = pd.value_counts(ld_info['block_idx']).sort_index().to_list()
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
        self.ld_info = self.ld_info.loc[self.ld_info['SNP'].isin(snps)]
        block_dict = {k: g["block_idx2"].tolist() for k,g in self.ld_info.groupby("block_idx")}
        self.block_sizes, self.block_ranges = self._get_block_info(self.ld_info)
        self.data = (block[block_dict[i]] for i, block in enumerate(self.data) if i in block_dict)

    
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
    def __init__(self, num_snps_part, ld_info, snp_getter, prop, inv=False):
        """
        Making an LD matrix with selected subjects and SNPs with an MAF > 0.01

        Parameters:
        ------------
        num_snps_part: a list of number of SNPs in each LD block
        ld_info: SNP information
        snp_getter: a generator for getting SNPs
        prop: proportion of variance to keep for each LD block
        inv: if take inverse or not 

        """
        self.data = []
        ldscore = []
        for _, num in enumerate(num_snps_part):
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
        ld_info['ldscore'] = np.concatenate(ldscore, axis=None)
        self.ld_info = ld_info

        
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
        block_avg = np.nanmean(block, axis = 0)
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
            prefix = f"{out}_ld_prop{int(prop*100)}"
        else:
            prefix = f"{out}_ld_inv_prop{int(prop*100)}"

        with open(f"{prefix}.ldmatrix", 'wb') as file:
            pickle.dump(self.data, file)
        self.ld_info.to_csv(f"{prefix}.ldinfo", sep='\t', index=None, header=None, float_format='%.3f')
        
        return prefix
    


def read_plink(dir, keep_snps=None, keep_indivs=None, maf=None):
    array_file, array_obj = f"{dir}.bed", PlinkBEDFile
    snp_file, snp_obj = f"{dir}.bim", PlinkBIMFile
    ind_file, ind_obj = f"{dir}.fam", PlinkFAMFile

    array_snps = snp_obj(snp_file)

    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)

    geno_array = array_obj(array_file, n, array_snps, keep_snps=keep_snps,
                           keep_indivs=keep_indivs, mafMin=maf)
    snp_getter = geno_array.nextSNPs
    array_snps.df['MAF'] = geno_array.df[:, 4]

    return array_snps.df, snp_getter



def partition_genome(bim, part, log):
    """
    # TODO: update block, is it correct that begin = -1 at the beginning?

    """
    num_snps_part = []
    end = -1
    bim['block_idx'] = None
    bim['block_idx2'] = None
    abs_begin = 0
    abs_end = 0
    n_skipped_blocks = 0
    block_idx = 0
    for i in range(part.shape[0]):
        cand = list(bim.loc[bim['CHR'] == part.iloc[i, 0], 'POS'])
        begin = end
        end = find_loc(cand, part.iloc[i, 2])
        if end < begin:
            begin = -1
        if end > begin:
            block_size = end - begin
            if block_size < 2000: 
                sub_blocks = [(begin, end)]
            else:
                log.info((f'A large LD block with size {block_size}, ',
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
                bim.loc[bim.index[abs_begin: abs_end], 'block_idx'] = block_idx
                bim.loc[bim.index[abs_begin: abs_end], 'block_idx2'] = range(sub_block_size)
                block_idx += 1
        else:
            n_skipped_blocks += 1
    log.info(f'{n_skipped_blocks} blocks with no SNP are skipped.')
    
    return num_snps_part, bim



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



# def check_partition(dir, log):
#     try:
#         header = open(dir).readline().split()
#     except:
#         log.info('ERROR: --partition should be an unzipped txt file')
#     if header[0] == 'X' or header[0] == '23':
#         raise ValueError('The X chromosome is not supported')
#     if len(header) != 3:
#         raise ValueError('The partition file should have three columns for CHR, START, and END (no headers)')
#     for x in header:
#         try:
#             int(x)
#         except:
#             log.info('ERROR: the CHR, START, and END should be an integer')



def check_input(args):
    ## required arguments
    if args.bfile is None:
        raise ValueError('--bfile is required.') 
    if args.partition is None:
        raise ValueError('--partition is required.')
    if args.ld_regu is None:
        raise ValueError('--ld-regu is required.')
    if args.out is None:
        raise ValueError('--out is required.')
    
    ## check file/directory exists
    dirname = os.path.dirname(args.out)
    if dirname is not None and not os.path.exists(os.path.dirname(args.out)):
        raise ValueError(f'{os.path.dirname(args.out)} does not exist.')
    
    if not os.path.exists(args.partition):
        raise ValueError(f'{args.partition} does not exist')
    try:
        header = open(dir).readline().split()
    except:
        raise ValueError('The genome partition file should be an unzipped txt file')
    for x in header[:3]:
        try:
            int(x)
        except:
            raise ValueError('The first three columns in --partition should be CHR, START, and END without header')

    ## processing some arguments
    try:
        ld_bfile, ld_inv_bfile = args.bfile.split(',')
    except:
        raise ValueError('Two bfiles should be provided with --bfile and separated with comma.')
    for suffix in ['bed', 'fam', 'bim']:
        if not os.path.exists(ld_bfile + suffix):
            raise ValueError(f'{ld_bfile + suffix} does not exist.')
        if not os.path.exists(ld_inv_bfile + suffix):
            raise ValueError(f'{ld_inv_bfile + suffix} does not exist.')
        
    try:
        ld_regu, ld_inv_regu = [float(x) for x in args.ld_regu.split(',')]
    except:
        raise ValueError('Two regularization levels should be provided with --prop and separated with comma.')
    if ld_regu >= 1 or ld_regu <= 0 or ld_inv_regu >= 1 or  ld_inv_regu <= 0:
        raise ValueError('Both regularization levels should be greater than 0 and less than 1.')


    return ld_bfile, ld_inv_bfile, ld_regu, ld_inv_regu



def run(args, log):
    ld_bfile, ld_inv_bfile, ld_prop, ld_inv_prop = check_input(args)

    log.info(f"Read SNP list from {ld_bfile}.bim and remove duplicated SNPs.")
    ld_bim = pd.read_csv(f'{ld_bfile}.bim', delim_whitespace=True, header=None,
                         names=['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2'])
    ld_bim.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
    log.info(f'{ld_bim.shape[0]} SNPs remaining for LD matrix.')

    log.info(f"Read SNP list from {ld_inv_bfile}.bim and remove duplicated SNPs.")
    ld_inv_bim = pd.read_csv(f'{ld_inv_bfile}.bim', delim_whitespace=True, header=None,
                             names=['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2'])
    ld_inv_bim.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
    log.info(f'{ld_inv_bim.shape[0]} SNPs remaining for LD inverse matrix.') 

    ld_merged = ld_bim.merge(ld_inv_bim, on='SNP')
    log.info((f"Merging SNP lists of LD matrix and its inverse, "
              f"{ld_merged.shape[0]} SNPs are common."))   
    
    log.info(f"Reading bfile from {ld_bfile} and keeping merged SNPs ...")
    ld_keep_snp_idx = ld_bim.loc[ld_bim['SNP'].isin(ld_merged['SNP'])].index.to_list()
    ld_bim, ld_snp_getter = read_plink(ld_bfile, ld_keep_snp_idx, )
    
    log.info(f"Reading bfile from {ld_inv_bfile} and keeping merged SNPs ...")
    ld_inv_keep_snp_idx = ld_inv_bim.loc[ld_inv_bim['SNP'].isin(ld_merged['SNP'])].index.to_list()
    ld_inv_bim, ld_inv_snp_getter = read_plink(ld_inv_bfile, ld_inv_keep_snp_idx, )

    log.info(f"Read genome partition info from {args.partition}.")
    # check_partition(args.partition)
    genome_part = pd.read_csv(args.partition, header=None, delim_whitespace=True)
    log.info(f"There are {genome_part.shape[0]} genome blocks to partition.")

    log.info(f"Doing genome partition ...")
    num_snps_part, ld_info = partition_genome(ld_bim, genome_part, log)
    log.info((f"There are {sum(num_snps_part)} SNPs partitioned into {len(num_snps_part)} blocks, "
              "with the biggest one {np.max(num_snps_part)} SNPs."))

    log.info('Making an LD matrix ...')
    ld = LDmatrixBED(num_snps_part, ld_info, ld_snp_getter, ld_prop)

    log.info('Making an LD inverse matrix ...')
    ld_inv = LDmatrixBED(num_snps_part, ld_info, ld_inv_snp_getter, ld_inv_prop, inv=True)
    
    ld_prefix = ld.save(args.out, False, ld_prop)
    log.info(f"Save LD matrix to {ld_prefix}.ldmatrix")
    log.info(f"Save LD matrix info to {ld_inv_prefix}.ldinfo")
    
    ld_inv_prefix = ld_inv.save(args.out, True, ld_inv_prop)
    log.info(f"Save LD inverse matrix to {ld_prefix}.ldmatrix")
    log.info(f"Save LD inverse matrix info to {ld_inv_prefix}.ldinfo")
