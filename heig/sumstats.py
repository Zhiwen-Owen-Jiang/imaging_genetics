import os
import re
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import utils

"""
required arguments:
--ldr-gwas: directory to LDR gwas files
--y2-gwas: directory to gwas file of a single trait
--out: directory to output
--n: sample size column
--snp: SNP column
--a1: A1 column
--a2: A2 column
--beta: beta column
--odds-ratio: odds_ratio column
--se: se column
--out: directory to save summmary statistics

optional arguments:
--maf: maf column
--info: info column
--maf-min: minimum maf
--info-min: minimum info score

"""

"""
TODO: debug

"""


def check_input(args):
    """
    Checking that all requirements are provided.
    Checking all arguments are valid.
    Replacing some args with the processed ones.
    
    """
    ## required arguments
    if args.ldr_gwas is None and args.y2_gwas is None:
        raise ValueError('Either --ldr-gwas or --y2-gwas should be provided.')
    if args.n is None :
        raise ValueError('--n is required.')
    if args.snp is None:
        raise ValueError('--snp is required.')
    if args.a1 is None:
        raise ValueError('--a1 is required.')
    if args.a2 is None:
        raise ValueError('--a2 is required.')
    if args.beta is None and args.odds_ratio is None:
        raise ValueError('Either --beta or --odds_ratio should be provided.')
    if args.se is None:
        raise ValueError('--se is required.')
    if args.out is None:
        raise ValueError('--out is required.')
    
    dirname = os.path.dirname(args.out)
    if dirname is not None and not os.path.exists(os.path.dirname(args.out)):
        raise ValueError(f'{os.path.dirname(args.out)} does not exist.')
    
    ## optional argument
    if args.maf is not None and args.maf_min is not None:
        try:
            args.maf_min = float(args.maf_min)
        except:
            raise ValueError('--maf-min should be a number.')
        if args.maf_min < 0 or args.maf_min > 1:
            raise ValueError('--maf-min should be between 0 and 1.')
    elif args.maf is None and args.maf_min:
        warnings.warn('No --maf column is provided. Ignore --maf-min')
        args.maf_min = None
    elif args.maf and args.maf_min is None:
        args.maf_min = 0.01
    
    if args.info is not None and args.info_min is not None:
        try:
            args.info_min = float(args.info_min)
        except:
            raise ValueError('--info-min should be a number.')
        if args.info_min < 0 or args.info_min > 1:
            raise ValueError('--info-min should be between 0 and 1')
    elif args.info is None and args.info_min:
        warnings.warn('No --info column is provided. Ignore --info-min')
        args.info_min = None
    elif args.info and args.info_min is None:
        args.info_min = 0.9

    ## processing some arguments
    
    
    if args.ldr_gwas:
        ldr_gwas_files = parse_gwas_input(args.ldr_gwas)
        for file in ldr_gwas_files:
            if not os.path.exists(file):
                raise ValueError(f"{file} does not exist.")
        args.gwas = ldr_gwas_files

    if args.y2_gwas:
        if not os.path.exists(args.y2_gwas):
            raise ValueError(f"{args.y2_gwas} does not exist.")
        args.gwas = [args.y2_gwas]

    return args



def parse_gwas_input(arg):
    """
    Parsing the LDR gwas files. 

    Parameters:
    ------------
    arg: the file name with indices, e.g., `results/hipp_left_f1_score.{0:19}.glm.linear`

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
        raise ValueError('--ldr-gwas should be specified using `{}`, e.g. `prefix.{stard:end}.suffix`')
    start, end = [int(x) for x in file_range.split(":")]
    gwas_files = [re.sub(p2, str(i), arg) for i in range(start, end + 1)]

    return gwas_files



def map_cols(args):
    """
    cols_map: keys are required columns, values are provided columns
    cols_map2: keys are provided columns, values are required columns

    """
    cols_map = dict()
    cols_map['N'] = args.n
    cols_map['SNP'] = args.snp
    cols_map['BETA'] = args.beta
    cols_map['SE'] = args.se
    cols_map['A1'] = args.a1
    cols_map['A2'] = args.a2
    cols_map['OR'] = args.odds_ratio
    cols_map['MAF'] = args.maf
    cols_map['MAF_MIN'] = args.maf_min
    cols_map['INFO'] = args.info
    cols_map['INFO_MIN'] = args.info_min
    
    cols_map2 = dict()
    for k, v in cols_map.items():
        if v is not None:
            cols_map2[v] = k

    return cols_map, cols_map2



def read_sumstats(dir):
    snpinfo_dir = f'{dir}.snpinfo'
    sumstats_dir = f'{dir}.sumstats'
    
    if not os.path.exists(snpinfo_dir) or not os.path.exists(sumstats_dir):
        raise ValueError(f"Either .sumstats or .snp file does not exist")
    
    sumstats = pickle.load(open(sumstats_dir, 'rb'))
    snpinfo = pd.read_csv(snpinfo_dir, delim_whitespace=True)
    
    if not snpinfo.shape[0] == sumstats.shape[0]:
        raise ValueError((f"Summary statistics and the meta data contain different number of SNPs, "
                          "which means the files have been modified."))
    
    return GWAS(sumstats['beta'], sumstats['se'], snpinfo)



class GWAS:
    required_cols = ['SNP', 'A1', 'A2', 'N', 'SE']
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'} 

    def __init__(self, beta, se, snpinfo):
        self.beta = beta
        self.se = se
        self.snpinfo = snpinfo


    @classmethod
    def from_rawdata(cls, gwas_files, cols_map, cols_map2, maf_min=None, info_min=None):
        """
        gwas_files: a list of gwas files
        cols_map: a dict mapping required colnames to provided colnames
        
        """
        cls.logger = logging.getLogger(__name__)
        r = len(gwas_files)
        
        for i, gwas_file in enumerate(gwas_files):
            openfunc, compression = utils.check_compression(gwas_file)
            cls._check_header(openfunc, compression, gwas_file, cols_map, cols_map2)
            gwas_data = pd.read_csv(gwas_file, delim_whitespace=True, compression=compression, 
                                    usecols=list(cols_map2.keys()), na_values=[-9, 'NONE']) # TODO: read by block
            gwas_data = gwas_data.rename(cols_map2, axis=1)

            if 'BETA' in cols_map:
                cls._check_median(gwas_data, 'BETA', 0)
            else:
                cls._check_median(gwas_data, 'OR', 1)
                gwas_data['BETA'] = np.log(gwas_data['OR'])
            
            gwas_data['A1'] = gwas_data['A1'].str.upper()
            gwas_data['A2'] = gwas_data['A2'].str.upper()

            if i == 0:
                orig_snps_list = gwas_data[['SNP', 'A1', 'A2', 'N']]
                beta_mat = np.zeros((gwas_data.shape[0], r))
                se_mat = np.zeros((gwas_data.shape[0], r))
                valid_snp_idxs = np.zeros((gwas_data.shape[0], r))
            else:
                if not gwas_data['SNP'].equals(orig_snps_list['SNP']):
                    raise ValueError('There are different SNPs in the input LDR GWAS files.')
            beta_mat[:, i] = np.array(gwas_data['BETA'])
            se_mat[:, i] = np.array(gwas_data['SE'])
            cls.logger.info(f'Pruning SNPs for {gwas_file} ...')
            gwas_data = cls._prune_snps(gwas_data, maf_min, info_min)
            final_snps_list = gwas_data['SNP']
            valid_snp_idxs[:, i] = orig_snps_list['SNP'].isin(final_snps_list)

        is_common_snp = (valid_snp_idxs == 1).all(axis=1)
        beta_mat = beta_mat[is_common_snp]
        se_mat = se_mat[is_common_snp]
        common_snp_info = orig_snps_list.loc[is_common_snp, ['SNP', 'A1', 'A2', 'N']]

        beta = pd.DataFrame(beta_mat, columns=[f'BETA{i}' for i in range(1, beta_mat.shape[1] + 1)], 
                            index=range(beta_mat.shape[0]))
        se = pd.DataFrame(se_mat, columns=[f'SE{i}' for i in range(1, se_mat.shape[1] + 1)], 
                            index=range(se_mat.shape[0]))
        snpinfo = common_snp_info.reset_index(drop=True)


        return cls(beta, se, snpinfo)

            
    def _prune_snps(self, gwas, maf_min, info_min):
        """
        Prune SNPs with 
        1) any missing values in required columns
        2) infinity in Z scores, less than 0 sample size
        3) any duplicates in rsID
        4) strand ambiguous
        5) an effective sample size less than 0.67 times the 90th percentage of sample size
        6) small MAF or small INFO score (optional)

        Parameters:
        ------------
        gwas: a pd.DataFrame of summary statistics with required columns
        maf_min: the minimum MAF
        info_min: the minimum INFO

        Returns:
        ---------
        A pd.DataFrame of pruned summary statistics

        """
        n_snps = self._check_ramaining_snps(gwas)
        self.logger.info(f"{n_snps} SNPs in the raw data.")

        gwas.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
        self.logger.info(f"Removed {n_snps - gwas.shape[0]} duplicated SNPs.")
        n_snps = self._check_ramaining_snps(gwas)

        gwas = gwas.loc[~gwas.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
        self.logger.info(f"Removed {n_snps - gwas.shape[0]} SNPs with any missing or infinite values.")
        n_snps = self._check_ramaining_snps(gwas)

        not_strand_ambiguous = [True if len(a2_) == 1 and len(a1_) == 1 and 
                                a2_ in self.complement and a1_ in self.complement and 
                                self.complement[a2_] != a1_ else False 
                                for a2_, a1_ in zip(gwas['A2'], gwas['A1'])]
        gwas = gwas.loc[not_strand_ambiguous]
        self.logger.info(f"Removed {n_snps - gwas.shape[0]} non SNPs and strand-ambiguous SNPs.")
        n_snps = self._check_ramaining_snps(gwas)

        n_thresh = int(gwas['N'].quantile(0.9) / 1.5)
        gwas = gwas.loc[gwas['N'] >= n_thresh]
        self.logger.info(f"Removed {n_snps - gwas.shape[0]} SNPs with N < {n_thresh}.")
        n_snps = self._check_ramaining_snps(gwas)

        if maf_min is not None:
            gwas = gwas.loc[gwas['MAF'] >= maf_min]
            self.logger.info(f"Removed {n_snps - gwas.shape[0]} SNPs with MAF < {maf_min}.")
            n_snps = self._check_ramaining_snps(gwas)

        if info_min is not None:
            gwas = gwas.loc[gwas['INFO'] >= info_min]
            self.logger.info(f"Removed {n_snps - gwas.shape[0]} SNPs with INFO < {info_min}.")
            n_snps = self._check_ramaining_snps(gwas)
        
        self.logger.info(f"{n_snps} SNPs remaining after pruning.\n")

        return gwas
    

    def _check_ramaining_snps(self, gwas):
        n_snps = gwas.shape[0]
        if n_snps == 0:
            raise ValueError('No SNP remaining. Check if misspecified columns.')
        return n_snps


    def extract_snps(self, keep_snps):
        if isinstance(keep_snps, pd.Series):
            keep_snps = pd.DataFrame(keep_snps, columns=['SNP'])
        self.snp_info['id'] = self.snp_info.index # keep the index in df
        self.snp_info = keep_snps.merge(self.snp_info, on='SNP')
        self.z_df = self.z_df.loc[self.snp_info['id']]
        del self.snp_info['id']
        

    def df2array(self):
        self.z_df = np.array(self.z_df)
        

    def _check_header(self, openfunc, compression, dir, cols_map, cols_map2):
        """
        First round check: if all required columns exist
        Second round check: if all provided columns exist

        """
        header = openfunc(dir).readline().split()
        if compression is not None:
            header[0] = str(header[0], 'UTF-8')
            header[1] = str(header[1], 'UTF-8')
        for col in self.required_cols:
            if cols_map[col] not in header:
                raise ValueError(f'{cols_map[col]} (case sensitive) cannot be found in {dir}.')
        for col, _ in cols_map2.items():
            if col not in header:
                raise ValueError(f'{col} (case sensitive) cannot be found in {dir}.')

        
    def _check_median(self, gwas_data, effect, null_value):
        median_beta = np.nanmedian(gwas_data[effect])
        if np.abs(median_beta - null_value > 0.1):
            raise ValueError((f"Median value of {effect} is {median_beta} (should be close to {null_value}). " 
                              "This column may be mislabeled."))
        else:
            self.logger.info(f"Median value of {effect} is {median_beta}, which is reasonable.")


    def get_zscore(self):
        self.z = self.beta / self.se
        

    def save(self, out):
        pickle.dump({'beta': self.beta, 'se': self.se}, open(f'{out}.sumstats', 'wb'), protocol=4)
        self.snpinfo.to_csv(f'{out}.snpinfo', sep='\t', index=None, na_rep='NA')

    


def run(args, log):
    args = check_input(args)
    cols_map, cols_map2 = map_cols(args)
    
    log.info(f'Reading and processing {len(args.gwas)} GWAS summary statistics ...')
    sumstats = GWAS.from_rawdata(args.gwas, cols_map, cols_map2, args.maf_min, args.info_min)
    sumstats.save(args.out)

    log.info(f'Save the processed summary statistics to {args.out}.sumstats and {args.out}.snpinfo')
