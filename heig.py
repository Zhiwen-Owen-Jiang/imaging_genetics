import os, time, argparse, traceback
import heig
from heig.utils import GetLogger, sec_to_str


VERSION = '1.0.0'

MASTHEAD = "******************************************************************************\n"
MASTHEAD += "* Highly-Efficient Imaging Genetics (HEIG)\n"
MASTHEAD += f"* Version {VERSION}\n"
MASTHEAD += f"* Zhiwen Jiang and Hongtu Zhu\n"
MASTHEAD += f"* Department of Biostatistics, University of North Carolina at Chapel Hill\n"
MASTHEAD += f"* GPL-3.0 licence\n"
MASTHEAD += f"* Correspondence: owenjf@live.unc.edu, zhiwenowenjiang@gmail.com\n"
MASTHEAD += "******************************************************************************\n"



parser = argparse.ArgumentParser(description='\n Highly-Efficient Imaging Genetics (HEIG)')
subparsers = parser.add_subparsers(dest='module', help='Choose an analysis module')

## global arguments
parser.add_argument('--out', help='directory for output')

## arguments for heri_gc.py
parser_heri_gc = subparsers.add_parser('heri-gc', 
                                       help='Heritability and (cross-trait) genetic correlation analysis')
parser_heri_gc.add_argument('--n-ldrs', type=int, 
                            help='number of LDRs to use')
parser_heri_gc.add_argument('--ldr-sumstats', 
                            help='directory to LDR summary statistics (prefix)')
parser_heri_gc.add_argument('--bases', 
                            help='directory to bases')
parser_heri_gc.add_argument('--inner-ldr', 
                            help='directory to inner product of LDR')
parser_heri_gc.add_argument('--ld-inv', 
                            help='directory to the inverse LD matrix')
parser_heri_gc.add_argument('--ld', 
                            help='directory to LD matrix')
parser_heri_gc.add_argument('--extract', 
                            help='a text file of SNPs to extract')
parser_heri_gc.add_argument('--y2-sumstats', 
                            help='gwas results of another traits')
parser_heri_gc.add_argument('--overlap', action='store_true', 
                            help='if there are sample overlap')
parser_heri_gc.add_argument('--heri-only', action='store_true', 
                            help='only compute heritability')

## arguments for ldr.py
parser_ldr = subparsers.add_parser('ldr', 
                                   help='LDR construction')
parser_ldr.add_argument('--image', 
                        help='directory to the imaging data')
parser_ldr.add_argument('--covar', 
                        help='directory to covariates')
parser_ldr.add_argument('--cat-covar-list', 
                        help='comma separated list of covariates to include in the analysis')
parser_ldr.add_argument('--coord', 
                        help='directory to coordinates')
parser_ldr.add_argument('--keep', 
                        help='a list of subjects to keep')
parser_ldr.add_argument('--prop', type=float, 
                        help='proportion of variance to keep, should be a number between 0 and 1.')
parser_ldr.add_argument('--all', action='store_true',
                        help=('if generating all components the selecting the top ones, '
                              'which may take longer time'))
parser_ldr.add_argument('--bw-opt', type=float, 
                        help=('the bandwidth you want to use, '
                              'then the program will skip searching the optimal bandwidth. '
                              'For images of any dimension, just specify one number, e.g, 0.5'))

## arguments for make_ld.py
parser_ld_matrix = subparsers.add_parser('ld-matrix', 
                                         help='Making LD matrix')
parser_ld_matrix.add_argument('--bfile', 
                              help=('directory to PLINK bfiles for LD matrix and its inverse. '
                                    'Only prefix is needed. Two prefices should be seperated by a comma'))
parser_ld_matrix.add_argument('--partition', 
                              help='directory to genome partition file')
parser_ld_matrix.add_argument('--ld-regu', 
                              help=('Regularization for LD matrix and its inverse. '
                                    'Two values should be separated by a comma. '
                                    'E.g., 0.85,0.80'))

## arguments for munge_sumstats.py
parser_sumstats = subparsers.add_parser('sumstats', 
                                        help='Organizing and preprocessing GWAS summary statistics')
parser_sumstats.add_argument('--ldr-gwas', 
                             help='directory to LDR gwas files (prefix)')
parser_sumstats.add_argument('--y2-gwas', 
                             help='directory to gwas file of a single trait')
parser_sumstats.add_argument('--n', 
                             help='sample size column')
parser_sumstats.add_argument('--snp', 
                             help='SNP column')
parser_sumstats.add_argument('--a1', 
                             help='A1 column')
parser_sumstats.add_argument('--a2', 
                             help='A2 column')
parser_sumstats.add_argument('--beta', 
                             help='beta column')
parser_sumstats.add_argument('--odds-ratio', 
                             help='odds ratio column')
parser_sumstats.add_argument('--se', 
                             help='se column')
parser_sumstats.add_argument('--maf', 
                             help='MAF column')
parser_sumstats.add_argument('--info', 
                             help='INFO column')
parser_sumstats.add_argument('--maf-min', type=float, 
                             help='minimum MAF')
parser_sumstats.add_argument('--info-min', type=float, 
                             help='minimum INFO')

## arguments for voxel_gwas.py
parser_voxelgwas = subparsers.add_parser('voxel-gwas', 
                                         help='Recovering voxel-level GWAS results')
parser_voxelgwas.add_argument('--n-ldrs', type=int, 
                              help='number of LDRs to use')
parser_voxelgwas.add_argument('--ldr-sumstats', 
                              help='directory to LDR gwas summary statistics (prefix)')
parser_voxelgwas.add_argument('--bases', 
                              help='directory to bases')
parser_voxelgwas.add_argument('--inner-ldr', 
                              help='directory to inner product of LDR')
parser_voxelgwas.add_argument('--sig-thresh', 
                              help='significance threshold for p-values, e.g. 0.00000005')
parser_voxelgwas.add_argument('--voxel', type=int, 
                              help='which voxel, 0 based index')
parser_voxelgwas.add_argument('--range',  
                              help=('a segment of chromosome, e.g. 3:1000000,3:2000000, '
                                    'where 3 is chromosome, and 1000000 is position'))
parser_voxelgwas.add_argument('--snp', 
                              help='which SNP to generate an atlas of association, rsID')



def main(args, log):
    if args.out is None:
        raise ValueError('--out is required.')
    if 'module' not in args:
        raise ValueError('Provide a module to tell HEIG which analysis to do.')
    if args.module == 'heri-gc':
        heig.heri_gc.run(args, log)
    elif args.module == 'ldr':
        heig.ldr.run(args, log)
    elif args.module == 'ld-matrix':
        heig.ldmatrix.run(args, log)
    elif args.module == 'sumstats':
        heig.sumstats.run(args, log)
    elif args.module == 'voxel-gwas':
        heig.voxel_gwas.run(args, log)
    else:
        raise ValueError(('Only the following modules are supported: '
                         'heri-gc, ldr, ld-matrix, sumstats, voxel-gwas'))
    


if __name__ == '__main__':
    args = parser.parse_args()

    logpath = os.path.join(f"{args.out}.heig")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "./heig.py\n"
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
