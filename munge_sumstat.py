import os, time, argparse, traceback
import re
import pickle
import gzip, bz2
import numpy as np
import pandas as pd
from numpy.linalg import eigh
from scipy.stats import norm, chi2
from utils import GetLogger, sec_to_str

MASTHEAD = "***********************************************************************************\n"
MASTHEAD += "* Organizing and formatting summary statistics\n"
MASTHEAD += "***********************************************************************************"


default_cnames = {
    # RS NUMBER
    'SNP': 'SNP',
    'MARKERNAME': 'SNP',
    'SNPID': 'SNP',
    'RS': 'SNP',
    'RSID': 'SNP',
    'RS_NUMBER': 'SNP',
    'RS_NUMBERS': 'SNP',
    # NUMBER OF STUDIES
    'NSTUDY': 'NSTUDY',
    'N_STUDY': 'NSTUDY',
    'NSTUDIES': 'NSTUDY',
    'N_STUDIES': 'NSTUDY',
    # P-VALUE
    'P': 'P',
    'PVALUE': 'P',
    'P_VALUE':  'P',
    'PVAL': 'P',
    'P_VAL': 'P',
    'GC_PVALUE': 'P',
    # ALLELE 1
    'A1': 'A1',
    'ALLELE1': 'A1',
    'ALLELE_1': 'A1',
    'EFFECT_ALLELE': 'A1',
    'REFERENCE_ALLELE': 'A1',
    'INC_ALLELE': 'A1',
    'EA': 'A1',
    # ALLELE 2
    'A2': 'A2',
    'ALLELE2': 'A2',
    'ALLELE_2': 'A2',
    'OTHER_ALLELE': 'A2',
    'NON_EFFECT_ALLELE': 'A2',
    'DEC_ALLELE': 'A2',
    'NEA': 'A2',
    # N
    'N': 'N',
    'NCASE': 'N_CAS',
    'CASES_N': 'N_CAS',
    'N_CASE': 'N_CAS',
    'N_CASES': 'N_CAS',
    'N_CONTROLS': 'N_CON',
    'N_CAS': 'N_CAS',
    'N_CON': 'N_CON',
    'N_CASE': 'N_CAS',
    'NCONTROL': 'N_CON',
    'CONTROLS_N': 'N_CON',
    'N_CONTROL': 'N_CON',
    'WEIGHT': 'N',  # metal does this. possibly risky.
    # SIGNED STATISTICS
    'ZSCORE': 'Z',
    'Z-SCORE': 'Z',
    'GC_ZSCORE': 'Z',
    'Z': 'Z',
    'OR': 'OR',
    'B': 'BETA',
    'BETA': 'BETA',
    'LOG_ODDS': 'LOG_ODDS',
    'EFFECTS': 'BETA',
    'EFFECT': 'BETA',
    'SIGNED_SUMSTAT': 'SIGNED_SUMSTAT',
    # INFO
    'INFO': 'INFO',
    # MAF
    'EAF': 'FRQ',
    'FRQ': 'FRQ',
    'MAF': 'FRQ',
    'FRQ_U': 'FRQ',
    'F_U': 'FRQ',
}

describe_cname = {
    'SNP': 'Variant ID (e.g., rs number)',
    'P': 'p-Value',
    'A1': 'Allele 1, interpreted as ref allele for signed sumstat.',
    'A2': 'Allele 2, interpreted as non-ref allele for signed sumstat.',
    'N': 'Sample size',
    'N_CAS': 'Number of cases',
    'N_CON': 'Number of controls',
    'Z': 'Z-score (0 --> no effect; above 0 --> A1 is trait/risk increasing)',
    'OR': 'Odds ratio (1 --> no effect; above 1 --> A1 is risk increasing)',
    'BETA': '[linear/logistic] regression coefficient (0 --> no effect; above 0 --> A1 is trait/risk increasing)',
    'LOG_ODDS': 'Log odds ratio (0 --> no effect; above 0 --> A1 is risk increasing)',
    'INFO': 'INFO score (imputation quality; higher --> better imputation)',
    'FRQ': 'Allele frequency',
    'SIGNED_SUMSTAT': 'Directional summary statistic as specified by --signed-sumstats.',
    'NSTUDY': 'Number of studies in which the SNP was genotyped.'
}

numeric_cols = ['P', 'N', 'N_CAS', 'N_CON', 'Z', 'OR', 'BETA', 'LOG_ODDS', 'INFO', 'FRQ', 'SIGNED_SUMSTAT', 'NSTUDY']



COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'} 



def parse_gwas_input(arg):
    """
    Parsing the LDR gwas files. 

    Parameters:
    ------------
    arg: the file name with indices, seperated by a comma, e.g., `results/hipp_left_f1_score.{0:19}.glm.linear`

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
        raise ValueError('--ldr-gwas should be specified using `{}`, \
                         e.g. `prefix.{stard:end}.suffix`')
    start, end = [int(x) for x in file_range.split(":")]
    gwas_files = [re.sub(p2, str(i), arg) for i in range(start, end + 1)]

    return gwas_files



def check_input(args):
    """
    Checking that all inputs are correct
    replacing some args with the processed ones
    currently (0313): n, snp, a1, a2, beta, se are required. Will be more flexible later
    
    """
    if not args.ldr_gwas and not args.y2_gwas:
        raise ValueError('Either --ldr-gwas or --y2-gwas should be provided')
    if not args.n:
        raise ValueError('--n is required')
    if not args.snp:
        raise ValueError('--snp is required')
    if not args.a1:
        raise ValueError('--a1 is required')
    if not args.a2:
        raise ValueError('--a2 is required')
    if not args.beta:
        raise ValueError('--beta is required')
    if not args.se:
        raise ValueError('--se is required')
    if not args.chr:
        raise ValueError('--chr is required')
    if not args.pos:
        raise ValueError('--pos is required')
    if not args.out:
        raise ValueError('--out is required')

    
    if not os.path.exists(os.path.dirname(args.out)):
        raise ValueError(f'{os.path.dirname(args.out)} does not exist')
    
    if args.ldr_gwas:
        ldr_gwas_files = parse_gwas_input(args.ldr_gwas)
        for file in ldr_gwas_files:
            if not os.path.exists(file):
                raise ValueError(f"{file} does not exist")
        args.gwas = ldr_gwas_files

    if args.y2_gwas:
        if not os.path.exists(args.y2_gwas):
            raise ValueError(f"{args.y2_gwas} does not exist")
        args.gwas = [args.y2_gwas]

    return args


def map_cols(args):
    """
    currently 

    """
    cols_map = dict()
    cols_map['N'] = args.n
    cols_map['SNP'] = args.snp
    cols_map['BETA'] = args.beta
    cols_map['SE'] = args.se
    cols_map['A1'] = args.a1
    cols_map['A2'] = args.a2
    cols_map['CHR'] = args.chr
    cols_map['POS'] = args.pos

    return cols_map




class GWAS:
    required_cols = ['SNP', 'A1', 'A2', 'N', 'BETA', 'SE']

    def __init__(self, gwas_files, cols_map):
        """
        gwas_files: a list of gwas files
        cols_map: a dict mapping required colnames to provided colnames
        
        """
        r = len(gwas_files)
        
        for i, gwas_file in enumerate(gwas_files):
            openfunc, compression = self._check_compression(gwas_file)
            self._check_header(openfunc, compression, gwas_file, cols_map)
            gwas_data = pd.read_csv(gwas_file, delim_whitespace=True, compression=compression, 
                                    usecols=list(cols_map.values()), na_values=[-9, 'NONE']) # TODO: read by block
            gwas_data = gwas_data.rename({cols_map['SNP']: 'SNP', cols_map['N']: 'N',
                                          cols_map['A1']: 'A1', cols_map['A2']: 'A2',
                                          cols_map['BETA']: 'BETA', cols_map['SE']: 'SE',
                                          cols_map['CHR']: 'CHR', cols_map['POS']: 'POS'}, axis=1)

            if i == 0:
                orig_snps_list = gwas_data[['CHR', 'SNP', 'POS', 'A1', 'A2', 'N']]
                beta_mat = np.zeros((gwas_data.shape[0], r))
                se_mat = np.zeros((gwas_data.shape[0], r))
                valid_snp_idxs = np.zeros((gwas_data.shape[0], r))
            else:
                if not gwas_data['SNP'].equals(orig_snps_list['SNP']):
                    raise ValueError('There are different SNPs in the input gwas files')
            beta_mat[:, i] = np.array(gwas_data['BETA'])
            se_mat[:, i] = np.array(gwas_data['SE'])
            log.info(f'Pruning SNPs for {gwas_file} ...')
            gwas_data = self._prune_snps(gwas_data, log)
            final_snps_list = gwas_data['SNP']
            valid_snp_idxs[:, i] = orig_snps_list['SNP'].isin(final_snps_list)

        is_common_snp = (valid_snp_idxs == 1).all(axis=1)
        beta_mat = beta_mat[is_common_snp]
        se_mat = se_mat[is_common_snp]
        common_snp_info = orig_snps_list.loc[is_common_snp, ['CHR', 'SNP', 'POS', 'A1', 'A2', 'N']]

        # self.z_mat = z_mat
        # self.SNP = common_snp_info['SNP']
        # self.A1 = common_snp_info['A1']
        # self.A2 = common_snp_info['A2']
        # self.N = common_snp_info['N']
        
        self.beta_df = pd.DataFrame(beta_mat, columns=[f'BETA{i}' for i in range(1, beta_mat.shape[1] + 1)], 
                            index=common_snp_info.index)
        self.se_df = pd.DataFrame(se_mat, columns=[f'SE{i}' for i in range(1, se_mat.shape[1] + 1)], 
                            index=common_snp_info.index)
        self.snp_info = common_snp_info

            
    def _prune_snps(self, gwas, log):
        """
        Prune SNPs with 
        1) any missing values in required columns
        2) infinity in Z scores, less than 0 sample size
        3) any duplicates in rsID
        4) strand ambiguous
        5) an effective sample size less than 0.67 times the 90th percentage of sample size

        Parameters:
        ------------
        gwas: a pd.DataFrame of summary statistics with required columns
        log: a logger

        Returns:
        ---------
        A pd.DataFrame of pruned summary statistics

        """
        n_snps = gwas.shape[0]
        log.info(f"{n_snps} SNPs in the raw data")

        gwas.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
        log.info(f"Removed {n_snps - gwas.shape[0]} duplicated SNPs")
        n_snps = gwas.shape[0]

        # gwas = gwas.loc[~gwas.isna().any(axis=1)]
        gwas = gwas.loc[~gwas.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
        log.info(f"Removed {n_snps - gwas.shape[0]} SNPs with any missing or infinite values")
        n_snps = gwas.shape[0]

        not_strand_ambiguous = [True if len(a2_) == 1 and len(a1_) == 1 and 
                                a2_ in COMPLEMENT and a1_ in COMPLEMENT and 
                                COMPLEMENT[a2_] != a1_ else False 
                                for a2_, a1_ in zip(gwas['A2'], gwas['A1'])]
        gwas = gwas.loc[not_strand_ambiguous]
        log.info(f"Removed {n_snps - gwas.shape[0]} non SNPs and strand-ambiguous SNPs")
        n_snps = gwas.shape[0]

        n_thresh = int(gwas['N'].quantile(0.9) / 1.5)
        gwas = gwas.loc[gwas['N'] >= n_thresh]
        log.info(f"Removed {n_snps - gwas.shape[0]} SNPs with N < {n_thresh}")
        
        n_snps = gwas.shape[0]
        log.info(f"{n_snps} SNPs remaining after pruning\n")

        return gwas
    

    def extract_snps(self, keep_snps):
        if isinstance(keep_snps, pd.Series):
            keep_snps = pd.DataFrame(keep_snps, columns=['SNP'])
        self.snp_info['id'] = self.snp_info.index # keep the index in df
        self.snp_info = keep_snps.merge(self.snp_info, on='SNP')
        self.z_df = self.z_df.loc[self.snp_info['id']]
        del self.snp_info['id']
        

    def df2array(self):
        self.z_df = np.array(self.z_df)
        

    def _check_compression(self, dir):
        """
        Checking which compression should use

        Parameters:
        ------------
        dir: diretory to the dataset

        Returns:
        ---------
        openfunc: function to open the file
        compression: type of compression
        
        """
        if dir.endswith('gz'):
            compression = 'gzip'
            openfunc = gzip.open
        elif dir.endswith('bz2'):
            compression = 'bz2'
            openfunc = bz2.BZ2File
        else:
            openfunc = open
            compression = None

        return openfunc, compression


    def _check_header(self, openfunc, compression, dir, cols_map):
        header = openfunc(dir).readline().split()
        if compression is not None:
            header[0] = str(header[0], 'UTF-8')
            header[1] = str(header[1], 'UTF-8')
        for col in self.required_cols:
            if cols_map[col] not in header:
                raise ValueError(f'{cols_map[col]} (case sensitive) cannot be found in {dir}')



def main(args, log):
    args = check_input(args)
    cols_map = map_cols(args)
    
    log.info(f'Reading and processing {len(args.gwas)} LDR gwas summary statistics ...')
    sumstat = GWAS(args.gwas, cols_map)

    pickle.dump(sumstat, open(f'{args.out}.sumstat2', 'wb'), protocol=4)
    log.info(f'Save the processed summary statistics to {args.out}.sumstat2')



parser = argparse.ArgumentParser()
parser.add_argument('--ldr-gwas', help='directory to LDR gwas files (prefix)')
parser.add_argument('--y2-gwas', help='directory to gwas file of a single trait')
parser.add_argument('--out', help='directory to output')

parser.add_argument('--n', help='sample size')
parser.add_argument('--snp', help='SNP column')
parser.add_argument('--a1', help='A1 column')
parser.add_argument('--a2', help='A2 column')
parser.add_argument('--p', help='p-value column')
parser.add_argument('--beta', help='beta column')
parser.add_argument('--or', help='or column')
parser.add_argument('--se', help='se column')
parser.add_argument('--chr', help='chr column')
parser.add_argument('--pos', help='pos column')


if __name__ == '__main__':
    args = parser.parse_args()

    logpath = os.path.join(f"{args.out}_munge_sumstat2.log")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "Parsed arguments\n"
        options = ['--'+x.replace('_','-')+' '+str(opts[x]) for x in non_defaults]
        header += '\n'.join(options).replace('True','').replace('False','')
        header = header+'\n'
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"Analysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")
