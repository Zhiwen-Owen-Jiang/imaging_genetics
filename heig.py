import os, time, argparse, traceback
import heig
from heig.utils import GetLogger, sec_to_str


VERSION = '1.0.0'

MASTHEAD = "******************************************************************************\n"
MASTHEAD += "* Highly-Efficient Imaging Genetics (HEIG)\n"
MASTHEAD += f"* Version {VERSION}\n"
MASTHEAD += f"* Zhiwen Jiang and Hongtu Zhu\n"
MASTHEAD += f"* Department of Biostatistics, University of North Carolina at Chapel Hill\n"
MASTHEAD += f"* GNU General Public License v3\n"
MASTHEAD += f"* Correspondence: owenjf@live.unc.edu, zhiwenowenjiang@gmail.com\n"
MASTHEAD += "******************************************************************************\n"



parser = argparse.ArgumentParser(description='\n Highly-Efficient Imaging Genetics (HEIG)')

## common arguments
parser.add_argument('--out', 
                    help='directory for output')
parser.add_argument('--n-ldrs', type=int, 
                    help='number of LDRs to use')
parser.add_argument('--ldr-sumstats', 
                    help='directory to LDR summary statistics (prefix)')
parser.add_argument('--bases', 
                    help='directory to bases')
parser.add_argument('--inner-ldr', 
                    help='directory to inner product of LDR')
parser.add_argument('--keep', 
                    help='a list of subjects to keep')

## arguments for heri_gc.py
parser.add_argument('--heri-gc', action='store_true',
                    help='Heritability and (cross-trait) genetic correlation analysis')
parser.add_argument('--ld-inv', 
                    help='directory to the inverse LD matrix')
parser.add_argument('--ld', 
                    help='directory to LD matrix')
parser.add_argument('--extract', 
                    help='a text file of SNPs to extract')
parser.add_argument('--y2-sumstats', 
                    help='gwas results of another traits')
parser.add_argument('--overlap', action='store_true', 
                    help='if there are sample overlap')
parser.add_argument('--heri-only', action='store_true', 
                    help='only compute heritability')

## arguments for ldr.py
parser.add_argument('--ldr', action='store_true',
                    help='LDR construction')
parser.add_argument('--image', 
                    help='directory to the imaging data')
parser.add_argument('--covar', 
                    help='directory to covariates')
parser.add_argument('--cat-covar-list', 
                    help='comma separated list of covariates to include in the analysis')
parser.add_argument('--coord', 
                    help='directory to coordinates')
parser.add_argument('--prop', type=float, 
                    help='proportion of variance to keep, should be a number between 0 and 1.')
parser.add_argument('--all', action='store_true',
                    help=('if generating all components the selecting the top ones, '
                          'which may take longer time'))
parser.add_argument('--bw-opt', type=float, 
                    help=('the bandwidth you want to use, '
                          'then the program will skip searching the optimal bandwidth. '
                          'For images of any dimension, just specify one number, e.g, 0.5'))

## arguments for make_ld.py
parser.add_argument('--ld-matrix', action='store_true',
                    help='Making LD matrix')
parser.add_argument('--bfile', 
                    help=('directory to PLINK bfiles for LD matrix and its inverse. '
                          'Only prefix is needed. Two prefices should be seperated by a comma'))
parser.add_argument('--partition', 
                    help='directory to genome partition file')
parser.add_argument('--ld-regu', 
                    help=('Regularization for LD matrix and its inverse. '
                          'Two values should be separated by a comma. '
                          'E.g., 0.85,0.80'))

## arguments for munge_sumstats.py
parser.add_argument('--sumstats', action='store_true',
                    help='Organizing and preprocessing GWAS summary statistics')
parser.add_argument('--ldr-gwas', 
                    help='directory to LDR gwas files (prefix)')
parser.add_argument('--y2-gwas', 
                    help='directory to gwas file of a single trait')
parser.add_argument('--n-col', 
                    help='sample size column')
parser.add_argument('--snp-col', 
                    help='SNP column')
parser.add_argument('--a1-col', 
                    help='A1 column')
parser.add_argument('--a2-col', 
                    help='A2 column')
parser.add_argument('--beta-col', 
                    help='beta column')
parser.add_argument('--odds-ratio-col', 
                    help='odds ratio column')
parser.add_argument('--se-col', 
                    help='se column')
parser.add_argument('--maf-col', 
                    help='MAF column')
parser.add_argument('--info-col', 
                    help='INFO column')
parser.add_argument('--maf-min', type=float, 
                    help='minimum MAF')
parser.add_argument('--info-min', type=float, 
                    help='minimum INFO')

## arguments for voxel_gwas.py
parser.add_argument('--voxel-gwas', action='store_true',
                    help='Recovering voxel-level GWAS results')
parser.add_argument('--sig-thresh', 
                    help='significance threshold for p-values, e.g. 0.00000005')
parser.add_argument('--voxel', type=int, 
                    help='which voxel, 0 based index')
parser.add_argument('--range',  
                    help=('a segment of chromosome, e.g. 3:1000000,3:2000000, '
                          'where 3 is chromosome, and 1000000 is position'))
parser.add_argument('--snp',
                    help='which SNP to generate an atlas of association, rsID')



def main(args, log):
    if args.out is None:
        raise ValueError('--out is required.')
    if args.heri_gc + args.ldr + args.ld_matrix + args.sumstats + args.voxel_gwas != 1:
        raise ValueError(('You must specify one and only one of following arguments: '
                          '--heri-gc, --ldr, --ld-matrix, --sumstats, --voxel-gwas.'))
    if args.heri_gc:
        heig.herigc.run(args, log)
    elif args.ldr:
        heig.ldr.run(args, log)
    elif args.ld_matrix:
        heig.ldmatrix.run(args, log)
    elif args.sumstats:
        heig.sumstats.run(args, log)
    elif args.voxel_gwas:
        heig.voxelgwas.run(args, log)
    


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
