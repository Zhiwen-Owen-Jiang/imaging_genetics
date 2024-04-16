import pickle
import re
import pandas as pd
import numpy as np


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

    # prefix, indices = arg.split(',')
    # min_index, max_index = [int(s) for s in indices.split('-')]
    # gwas_files = [f"{prefix}.{i}.glm.linear" for i in range(min_index, max_index + 1)]
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
        # if not (np.diff(pd.unique(self.ld_info['CHR'])) > 0).all():
        #     raise ValueError(f'The chrs in the LD matrix are not sorted')
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
            # if ld_i.ld_info.loc[ld_i.ld_info.index[0], 'CHR'] >= self.ld_info.loc[self.ld_info.index[-1], 'CHR']:
            #     raise ValueError('Can only merge LD matrices in order (chr1, chr2, ...)')
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
            # while True:
            #     try:
            #         yield pickle.load(file)
            #     except EOFError:
            #         break


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


    # def _get_2d_matrix(self, dim, tril):
    #     """
    #     Recovering the lower triangle w/o the diag of a matrix

    #     """
    #     if len(tril.shape) == 2:
    #         raise ValueError('The block has wrong dimension (data may have been truncated)')
    #     matrix = np.zeros((dim, dim))
    #     matrix[np.tril_indices(matrix.shape[0], k = -1)] = tril
    #     matrix = matrix + matrix.T
    #     np.fill_diagonal(matrix, 1)

    #     return matrix


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
        # self.data = self._update_generator_extraction(block_dict)
        self.data = (block[block_dict[i]] for i, block in enumerate(self.data) if i in block_dict)

        
    # def _update_generator_extraction(self, block_dict):
    #     """
    #     Updating each LD block in extraction

    #     """
    #     for i, block in enumerate(self.data):
    #         if i not in block_dict:
    #             continue
    #         else:
    #             # yield self._extract_from_eigen(block, block_dict[i])
    #             yield block[block_dict[i]]
                

    # def _extract_from_eigen(self, block, idxs):
    #     """
    #     Extracting SNPs from an LD block (eigenvalues, eigenvectors) using indices

    #     Parameters:
    #     ------------
    #     data: a tuple (eigenvalues, eigenvectors)
    #     idxs: idxs to keep

    #     """
    #     block = block[idxs]
        
    #     return block

 
    # def _extract_from_tril(self, data, size, idxs):
    #     """
    #     Extracting SNPs from an LD block using indices

    #     Parameters:
    #     ------------
    #     data: lower triangle w/o diag of an LD block
    #     size: original size of the LD block
    #     idxs: cols and rows to keep in the original LD block

    #     Returns:
    #     ---------
    #     The extracted lower triangle w/o diag of the LD block

    #     """
    #     cum_sum = np.cumsum(range(size - 1))
    #     idx = [cum_sum[idxs[i] - 1] + idxs[:i] for i in range(1, len(idxs))]
    #     if len(idx) > 0:
    #         idx = np.concatenate(idx)
    #         return data[idx] 
    #     else:
    #         return np.array([])
    
    
    # def truncate(self, prop, inv):
    #     """
    #     Truncating SNPs based on eigenvalues
    
    #     Parameters:
    #     ------------
    #     prop: the proportion of variance to keep
    #     inv: if inverse the LD matrix  

    #     Returns:
    #     ---------
    #     A block diagonal matrix

    #     """
        
    #     if prop == 1:
    #         for i, block in enumerate(self.data):
    #             yield self._get_2d_matrix(self.block_sizes[i], block)
    #     else: 
    #         for i, block in enumerate(self.data):
    #             block = self._get_2d_matrix(self.block_sizes[i], block)
    #             values, bases = np.linalg.eigh(block)
    #             values = np.flip(values)
    #             bases = np.flip(bases, axis=1)
    #             prop_var = np.cumsum(values) / np.sum(values)
    #             idxs = (prop_var <= prop) & (values != 0)
    #             values = values[idxs]
    #             bases = bases[:, idxs]
    #             if inv:
    #                 yield np.dot(bases * values ** -1, bases.T)
    #             else:
    #                 yield np.dot(bases * values, bases.T)
        
        

    # def estimate_ldscore(self):
    #     """
    #     Estimating LD score from the LD matrix
    #     The Pearson correlation is adjusted by r2 - (1 - r2) / (N - 2)

    #     """
    #     if not self.is_truncated:
    #         raise ValueError('Truncate the LD matrix first then estimate LD scores')
    #     n_samples = np.array(self.ld_info['N'])
    #     # if n_samples < 3:
    #     #     raise ValueError('The number of samples to estimate LD matrix is wrong')
    #     ldscore = np.zeros(self.ld_info.shape[0])
    #     for i, (begin, end) in enumerate(self.block_ranges):
    #         block = self.data[i]
    #         raw_ld = np.sum(block ** 2, axis=0)
    #         adj_ld = raw_ld - (1 - raw_ld) / (n_samples[i] - 2) 
    #         ldscore[begin: end] = adj_ld

    #     merged_blocks = self._merge_blocks(self.block_sizes)

    #     return ldscore, merged_blocks
    
    
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
        TODO: remove all duplicated SNPs, add a column for MAF

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
        for i, num in enumerate(num_snps_part):
            # print(f'Computing LD block{i+1} with size {num}*{num} ...')
            block = snp_getter(num)
            block = self._fill_na(block)
            corr = np.atleast_2d(np.corrcoef(block.T))
            ldscore_i = self._estimate_ldscore(corr, block.shape[0])
            ldscore.append(ldscore_i)
            # tril_corr = self._get_lower_triangle(corr)
            # self.data.append(tril_corr)
            values, bases = self._truncate(corr, prop)
            if inv:
                bases = bases * np.sqrt(values ** -1)
            else:
                bases = bases * np.sqrt(values)
            self.data.append(bases)
        ld_info['ldscore'] = np.concatenate(ldscore, axis=None)
        self.ld_info = ld_info

        # save the data as a generator
        # self.data = (x for x in self.data)

        
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
    
    
    # def _get_lower_triangle(self, matrix):
    #     """
    #     Extracting only the lower triangle w/o the diag of a matrix

    #     """
    #     return matrix[np.tril_indices(matrix.shape[0], k = -1)]
    

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