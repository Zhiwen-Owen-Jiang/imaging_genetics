import os, time, argparse, traceback, numexpr
import heig.sumstats as sumstats
import heig.herigc as herigc
import heig.ksm as ksm
import heig.fpca as fpca
import heig.ldmatrix as ldmatrix
import heig.voxelgwas as voxelgwas
from heig.utils import GetLogger, sec_to_str


os.environ['NUMEXPR_MAX_THREADS'] = '8'
numexpr.set_num_threads(int(os.environ['NUMEXPR_MAX_THREADS']))


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

common_parser = parser.add_argument_group(title="Common arguments")
herigc_parser = parser.add_argument_group(title="Arguments specific to heritability and (cross-trait) genetic correlation analysis")
ksm_parser = parser.add_argument_group(title="Arguments specific to Kernel smoothing")
fpca_parser = parser.add_argument_group(title="Arguments specific to functional PCA")
makeld_parser = parser.add_argument_group(title="Arguments specific to making an LD matrix and its inverse")
sumstats_parser = parser.add_argument_group(title="Arguments specific to organizing and preprocessing GWAS summary statistics")
voxelgwas_parser = parser.add_argument_group(title="Arguments specific to recovering voxel-level GWAS results")

## module arguments
herigc_parser.add_argument('--heri-gc', action='store_true',
                           help='Heritability and (cross-trait) genetic correlation analysis.')
ksm_parser.add_argument('--kernel-smooth', action='store_true',
                        help='Kernel smoothing.')
fpca_parser.add_argument('--fpca', action='store_true',
                         help='Functional PCA.')
makeld_parser.add_argument('--ld-matrix', action='store_true',
                           help='Making an LD matrix and its inverse.')
sumstats_parser.add_argument('--sumstats', action='store_true',
                             help='Organizing and preprocessing GWAS summary statistics.')
voxelgwas_parser.add_argument('--voxel-gwas', action='store_true',
                              help='Recovering voxel-level GWAS results.')

## common arguments
common_parser.add_argument('--out', 
                           help='Prefix of output.')
common_parser.add_argument('--n-ldrs', type=int, 
                           help='Number of LDRs. Supported modules: --heri-gc, --fpca, --voxel-gwas.')
common_parser.add_argument('--ldr-sumstats', 
                           help=('Prefix of preprocessed LDR GWAS summary statistics. '
                                 'Supported modules: --heri-gc, --voxel-gwas.'))
common_parser.add_argument('--bases', 
                           help='Directory to functional bases. Supported modules: --heri-gc, --voxel-gwas.')
common_parser.add_argument('--inner-ldr', 
                           help='Directory to inner product of LDRs. Supported modules: --heri-gc, --voxel-gwas.')
common_parser.add_argument('--keep', 
                           help=('File with individual IDs to keep in the kernel smoothing and functional PCA. '
                                 'The file should be tab or space delimited without header, '
                                 'with the first column being FID and the second column being IID. '
                                 'Other columns in the file will be ignored. '
                                 'Each row contains only one subject. '
                                 'Supported modules: --kernel-smooth, --fpca, --make-ld.'))
common_parser.add_argument('--extract', 
                           help=('File with SNPs to include in the LD matrix and its inverse and in voxelwise GWAS. '
                                 'The file should be tab or space delimited without header, '
                                 'with the first column being rsID. '
                                 'Other columns in the file will be ignored. '
                                 'Each row contains only one subject. '
                                 'Supported modules: --heri-gc, --make-ld, --voxel-gwas.'))
common_parser.add_argument('--maf-min', type=float, 
                             help=('Minimum minor allele frequency for screening SNPs. '
                                   'Supported modules: --make-ld, --sumstats.'))

## arguments for heri_gc.py
herigc_parser.add_argument('--ld-inv', 
                           help=('Prefix of inverse LD matrix. Multiple matrices can be specified using {}, '
                                 'e.g., `ld_inv_chr{1:22}`.'))
herigc_parser.add_argument('--ld', 
                           help=('Prefix of LD matrix. Multiple matrices can be specified using {}, '
                                 'e.g., `ld_chr{1:22}`.'))
herigc_parser.add_argument('--y2-sumstats', 
                           help='Prefix of preprocessed GWAS summary statistics of non-imaging traits.')
herigc_parser.add_argument('--overlap', action='store_true', 
                           help=('Flag for indicating sample overlap between LDR summary statistics '
                                 'and non-imaging summary statistics. Only effective if --y2-sumstats is specified.'))
herigc_parser.add_argument('--heri-only', action='store_true', 
                           help=('Flag for only computing voxelwise heritability '
                                 'but not voxelwise genetic correlation within images.'))

## arguments for ksm.py
ksm_parser.add_argument('--image-dir', 
                        help=('Directory to images. All images in the directory with matched suffix '
                              '(see --image-suffix) and will be loaded. '
                              'Multiple directories can be specified and separated by comma. '
                              '--keep can be used to load a subset of images. '
                              'The supported formats include all those that can be loaded by '
                              'nibabel.load(), such as .nii, .nii.gz, .mgh, .mgz, etc. '
                              'And FreeSurfer morphometry data file.'))
ksm_parser.add_argument('--image-suffix', 
                        help=('Suffix of images. HEIG requires the name of each image being <ID><suffix>, '
                              'e.g., `1000001_masked_FAskel.nii.gz`, where `1000001` is the ID '
                              'and `_masked_FAskel.nii.gz` is the suffix. '
                              'HEIG will collect ID for each image. '
                              'Multiple suffixes can be specified and separated by comma.'))
ksm_parser.add_argument('--surface-mesh',
                        help=('Directory to FreeSurfer surface mesh data. '
                              'Required if loading FreeSurfer morphometry data files.'))
ksm_parser.add_argument('--bw-opt', type=float, 
                        help=('The bandwidth you want to use, '
                              'then the program will skip searching the optimal bandwidth. '
                              'For images of any dimension, just specify one number, e.g, 0.5'))

## arguments for fpca.py
fpca_parser.add_argument('--image', 
                         help='Directory to processed raw images in HDF5 format.')
fpca_parser.add_argument('--sm-image', 
                         help='Directory to processed smoothed images in HDF5 format.')
fpca_parser.add_argument('--covar', 
                         help='Directory to covariate file.')
fpca_parser.add_argument('--cat-covar-list', 
                         help='List of categorical covariates to include in the analysis. '
                         'Each covariate is separated by a comma.')
fpca_parser.add_argument('--prop', type=float, 
                         help='Proportion of imaging signals to keep, must be a number between 0 and 1.')
fpca_parser.add_argument('--all', action='store_true', 
                         help=('Flag for generating all components which is min(n_subs, n_voxels), '
                               'which may take longer time very memory consuming.'))

## arguments for make_ld.py
makeld_parser.add_argument('--bfile', 
                           help=('Prefix of PLINK bfile triplets for LD matrix and its inverse. '
                                 'Two prefices should be seperated by a comma, e.g., `file1,file2` .'))
makeld_parser.add_argument('--partition', 
                           help=('Genome partition file. '
                                 'The file should be tab or space delimited without header, '
                                 'with the first column being chromosome, '
                                 'the second column being the start position, '
                                 'and the third column being the end position.'
                                 'Each row contains only one LD block.'))
makeld_parser.add_argument('--ld-regu',
                           help=('Regularization for LD matrix and its inverse. '
                                 'Two values should be separated by a comma, '
                                 'e.g., `0.85,0.80`'))

## arguments for munge_sumstats.py
sumstats_parser.add_argument('--ldr-gwas', 
                             help=('Raw LDR GWAS summary statistics. '
                                   'Multiple files can be speficied using {:}, e.g., `ldr_gwas{1:10}.txt`'))
sumstats_parser.add_argument('--y2-gwas', 
                             help='Raw non-imaging GWAS summary statistics.')
sumstats_parser.add_argument('--n', type=int, 
                             help='Sample size. A positive number.')
sumstats_parser.add_argument('--n-col', 
                             help='Sample size column.')
sumstats_parser.add_argument('--chr-col', 
                             help='Chromosome column.')
sumstats_parser.add_argument('--pos-col', 
                             help='Position column.')
sumstats_parser.add_argument('--snp-col', 
                             help='SNP column.')
sumstats_parser.add_argument('--a1-col', 
                             help='A1 column. The effective allele.')
sumstats_parser.add_argument('--a2-col', 
                             help='A2 column. The non-effective allele.')
sumstats_parser.add_argument('--effect-col', 
                             help=('Genetic effect column, usually beta or odds ratio, '
                                   'should be specified in this format `BETA,0` where '
                                   'BETA is the column name and 0 is the null value. '
                                   'For odds ratio, the null value is 1.'))
sumstats_parser.add_argument('--se-col', 
                             help='Standard error column.')
sumstats_parser.add_argument('--z-col', 
                             help='Z score column.')
sumstats_parser.add_argument('--p-col', 
                             help='p-Value column.')
sumstats_parser.add_argument('--maf-col', 
                             help='Minor allele frequency column.')
sumstats_parser.add_argument('--info-col', 
                             help='INFO score column.')
sumstats_parser.add_argument('--info-min', type=float, 
                             help='Minimum INFO score for screening SNPs.')

## arguments for voxel_gwas.py
voxelgwas_parser.add_argument('--sig-thresh', type=float, 
                              help=('Significance p-value threshold, '
                                    'can be specified in a decimal 0.00000005 '
                                    'or in scientific notation 5e-08'))
voxelgwas_parser.add_argument('--voxel', type=int, 
                              help='0 based index of voxel.')
voxelgwas_parser.add_argument('--range', 
                              help=('A segment of chromosome, e.g. `3:1000000,3:2000000` '
                                    'means from chromosome 3 bp 1000000 to chromosome 3 bp 2000000. '
                                    'Cross chromosome is not allowed.'))



def main(args, log):
    if args.out is None:
        args.out = 'heig'
    else:
        dirname = os.path.dirname(args.out)
        if dirname != '' and not os.path.exists(dirname):
            raise ValueError(f'{os.path.dirname(args.out)} does not exist.')
    if args.heri_gc + args.kernel_smooth + args.fpca + args.ld_matrix + args.sumstats + args.voxel_gwas != 1:
        raise ValueError(('You must specify one and only one of following arguments: '
                          '--heri-gc, --kernel-smooth, --fpca, --ld-matrix, --sumstats, --voxel-gwas.'))
    if args.heri_gc:
        herigc.run(args, log)
    elif args.kernel_smooth:
        ksm.run(args, log)
    elif args.fpca:
        fpca.run(args, log)
    elif args.ld_matrix:
        ldmatrix.run(args, log)
    elif args.sumstats:
        sumstats.run(args, log)
    elif args.voxel_gwas:
        voxelgwas.run(args, log)
    


if __name__ == '__main__':
    args = parser.parse_args()

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "heig.py \\\n"
        options = ['--'+x.replace('_','-')+' '+str(opts[x]) + ' \\'  for x in non_defaults]
        header += '\n'.join(options).replace('True','').replace('False','')
        header = header+'\n'
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"\nAnalysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")
