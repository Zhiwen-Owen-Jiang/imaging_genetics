import os, time, argparse, traceback
from heig.utils import GetLogger, sec_to_str


MASTHEAD = ''


def main(args, log):
    pass




parser = argparse.ArgumentParser()

## arguments for heri_gc.py
parser.add_argument('--heri-gc')
parser.add_argument('--n-ldrs', type=int, help='number of LDRs to use')
parser.add_argument('--ldr-gwas', help='directory to LDR gwas files (prefix)')
parser.add_argument('--bases', help='directory to bases')
parser.add_argument('--inner-ldr', help='directory to inner product of LDR')
parser.add_argument('--ld-inv', help='directory to the inverse LD matrix')
parser.add_argument('--ld', help='directory to LD matrix')
parser.add_argument('--out', help='directory for output')
parser.add_argument('--extract', help='a text file of SNPs to extract')
parser.add_argument('--y2-gwas', help='gwas results of another traits')
parser.add_argument('--overlap', action='store_true', help='if there are sample overlap')
parser.add_argument('--heri-only', action='store_true', help='only compute heritability')


## arguments for ldr.py
parser.add_argument('--ldr')
parser.add_argument('--image', help='directory to the imaging data')
parser.add_argument('--covar', help='directory to covariates')
parser.add_argument('--cat-covar-list', help='comma separated list of covariates to include in the analysis')
parser.add_argument('--coord', help='directory to coordinates')
parser.add_argument('--out', help='prefix for output')
parser.add_argument('--keep', help='a list of subjects to keep')
parser.add_argument('--var-keep', type=float, help='proportion of variance to keep, should be a number between 0 and 1.')
parser.add_argument('--all', action='store_true', help='if generating all components the selecting the top ones, which may take longer time')
parser.add_argument('--bw-opt', type=float, help='the bandwidth you want to use, \
                    then the program will skip searching the optimal bandwidth. \
                    For images of any dimension, just specify one number, e.g, 0.5')

## arguments for make_ld.py
parser.add_argument('--make-ld-matrix')


## arguments for munge_sumstats.py
parser.add_argument('--munge-sumstats')


if __name__ == '__main__':
    args = parser.parse_args()

    logpath = os.path.join(f"{args.out}_heig.log")
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
