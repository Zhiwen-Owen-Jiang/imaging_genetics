import os
import time
import argparse
import traceback
import numexpr
from heig.utils import GetLogger, sec_to_str


os.environ['NUMEXPR_MAX_THREADS'] = '8'
numexpr.set_num_threads(int(os.environ['NUMEXPR_MAX_THREADS']))


VERSION = '1.1.0'
MASTHEAD = "******************************************************************************\n"
MASTHEAD += "* Highly-Efficient Imaging Genetics (HEIG)\n"
MASTHEAD += f"* Version {VERSION}\n"
MASTHEAD += f"* Zhiwen Jiang and Hongtu Zhu\n"
MASTHEAD += f"* Department of Biostatistics, University of North Carolina at Chapel Hill\n"
MASTHEAD += f"* GNU General Public License v3\n"
MASTHEAD += f"* Correspondence: owenjf@live.unc.edu, zhiwenowenjiang@gmail.com\n"
MASTHEAD += "******************************************************************************\n"


parser = argparse.ArgumentParser(
    description='\n Highly-Efficient Imaging Genetics (HEIG) v1.1.0')

common_parser = parser.add_argument_group(title="Common arguments")
herigc_parser = parser.add_argument_group(
    title="Arguments specific to heritability and (cross-trait) genetic correlation analysis")
ksm_parser = parser.add_argument_group(
    title="Arguments specific to Kernel smoothing")
fpca_parser = parser.add_argument_group(
    title="Arguments specific to functional PCA")
makeld_parser = parser.add_argument_group(
    title="Arguments specific to making an LD matrix and its inverse")
sumstats_parser = parser.add_argument_group(
    title="Arguments specific to organizing and preprocessing GWAS summary statistics")
voxelgwas_parser = parser.add_argument_group(
    title="Arguments specific to recovering voxel-level GWAS results")
gwas_parser = parser.add_argument_group(
    title="Arguments specific to doing genome-wide association analysis")
annot_vcf_parser = parser.add_argument_group(
    title="Arguments specific to annotating VCF files")
wgs_null_parser = parser.add_argument_group(
    title="Arguments specific to the null model of whole genome sequencing analysis")
wgs_coding_parser = parser.add_argument_group(
    title='Arguments specific to whole genome sequencing analysis for coding variants')

# module arguments
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
gwas_parser.add_argument('--gwas', action='store_true',
                         help='Doing genome-wide association analysis.')
annot_vcf_parser.add_argument('--annot-vcf', action='store_true',
                              help='Annotating VCF files.')
wgs_null_parser.add_argument('--wgs-null', action='store_true',
                        help='Fitting the null model of whole genome sequencing analysis.')
wgs_coding_parser.add_argument('--wgs-coding', action='store_true',
                        help='Whole genome sequencing analysis for coding variants.')

# common arguments
common_parser.add_argument('--out',
                           help='Prefix of output.')
common_parser.add_argument('--n-ldrs', type=int,
                           help=('Number of LDRs. Supported modules: '
                                 '--heri-gc, --fpca, --voxel-gwas, --wgs-null, --wgs-coding.'))
common_parser.add_argument('--ldr-sumstats',
                           help=('Prefix of preprocessed LDR GWAS summary statistics. '
                                 'Supported modules: --heri-gc, --voxel-gwas.'))
common_parser.add_argument('--bases',
                           help=('Directory to functional bases. Supported modules: '
                                 '--heri-gc, --voxel-gwas, --wgs-null.'))
common_parser.add_argument('--inner-ldr',
                           help=('Directory to inner product of LDRs. '
                                 'Supported modules: --heri-gc, --voxel-gwas.'))
common_parser.add_argument('--keep',
                           help=('Individual file(s). Multiple files are separated by comma. '
                                 'Each file should be tab or space delimited, '
                                 'with the first column being FID and the second column being IID. '
                                 'Other columns will be ignored. '
                                 'Each row contains only one subject. '
                                 'Supported modules: --kernel-smooth, --fpca, --ld-matrix, '
                                 '--wgs-null, --wgs-coding.'))
common_parser.add_argument('--extract',
                           help=('SNP file(s). Multiple files are separated by comma. '
                                 'Each file should be tab or space delimited, '
                                 'with the first column being rsID. '
                                 'Other columns will be ignored. '
                                 'Each row contains only one SNP. '
                                 'Supported modules: --heri-gc, --ld-matrix, --voxel-gwas, '
                                 '--wgs-coding.'))
common_parser.add_argument('--maf-min', type=float,
                           help=('Minimum minor allele frequency for screening SNPs. '
                                 'Supported modules: --ld-matrix, --sumstats, '
                                 '--wgs-coding.'))
common_parser.add_argument('--covar',
                           help=('Directory to covariate file. '
                                 'Supported modules: --fpca, --gwas, --wgs-null.'))
common_parser.add_argument('--cat-covar-list',
                           help=('List of categorical covariates to include in the analysis. '
                                 'Multiple covariates are separated by comma. '
                                 'Supported modules: --fpca, --gwas, --wgs-null.'))
common_parser.add_argument('--bfile',
                           help=('Prefix of PLINK bfile triplets. '
                                 'When estimating LD matrix and its inverse, two prefices should be provided '
                                 'and seperated by a comma, e.g., `prefix1,prefix2`. '
                                 'When doing GWAS, only one prefix is allowed. '
                                 'Supported modules: --ld-matrix, --gwas.'))
common_parser.add_argument('--range',
                           help=('A segment of chromosome, e.g. `3:1000000,3:2000000`, '
                                 'from chromosome 3 bp 1000000 to chromosome 3 bp 2000000. '
                                 'Cross-chromosome is not allowed. And the end position must '
                                 'be greater than the start position. '
                                 'Supported modules: --voxel-gwas, --wgs-coding.'))
common_parser.add_argument('--voxel',
                              help=('one-based index of voxel or a file containing voxels. '
                                    'Supported modules: --voxel-gwas, --wgs-coding.'))
common_parser.add_argument('--ldrs',
                           help='Directory to LDR file. Supported modules: --gwas, --wgs-null.')
common_parser.add_argument('--geno-mt',
                           help='Directory to genotype MatrixTable. Supported modules: --gwas, --wgs-coding.')
common_parser.add_argument('--grch37', action='store_true',
                           help=('Using reference genome GRCh37. Otherwise using GRCh38. '
                                 'Supported modules: --gwas, --annot-vcf.'))
common_parser.add_argument('--not-save-genotype-data', action='store_true',
                           help=('Do not save preprocessed genotype data. '
                                 'Supported modules: --gwas, --wgs-coding.'))

# arguments for herigc.py
herigc_parser.add_argument('--ld-inv',
                           help=('Prefix of inverse LD matrix. Multiple matrices can be specified using {:}, '
                                 'e.g., `ld_inv_chr{1:22}_unrel`.'))
herigc_parser.add_argument('--ld',
                           help=('Prefix of LD matrix. Multiple matrices can be specified using {:}, '
                                 'e.g., `ld_chr{1:22}_unrel`.'))
herigc_parser.add_argument('--y2-sumstats',
                           help='Prefix of preprocessed GWAS summary statistics of non-imaging traits.')
herigc_parser.add_argument('--overlap', action='store_true',
                           help=('Flag for indicating sample overlap between LDR summary statistics '
                                 'and non-imaging summary statistics. Only effective if --y2-sumstats is specified.'))
herigc_parser.add_argument('--heri-only', action='store_true',
                           help=('Flag for only computing voxelwise heritability '
                                 'and skipping voxelwise genetic correlation within images.'))

# arguments for ksm.py
ksm_parser.add_argument('--image-dir',
                        help=('Directory to images. All images in the directory with matched suffix '
                              '(see --image-suffix) will be loaded. '
                              'Multiple directories can be provided and separated by comma. '
                              '--keep can be used to load a subset of images (see --keep). '
                              'The supported formats include NIFTI and CIFTI images '
                              'and FreeSurfer morphometry data file.'))
ksm_parser.add_argument('--image-suffix',
                        help=('Suffix of images. HEIG requires the name of each image in the format <ID><suffix>, '
                              'e.g., `1000001_masked_FAskel.nii.gz`, where `1000001` is the ID '
                              'and `_masked_FAskel.nii.gz` is the suffix. '
                              'HEIG will collect ID for each image. '
                              'Multiple suffixes can be specified and separated by comma '
                              'and the number of directories must match the number of suffices.'))
ksm_parser.add_argument('--surface-mesh',
                        help=('Directory to FreeSurfer surface mesh data. '
                              'Required if loading FreeSurfer morphometry data files.'))
ksm_parser.add_argument('--gifti',
                        help=('Directory to GIFTI data for surface geometry. '
                              'Required if loading CIFTI2 surface data.'))
ksm_parser.add_argument('--bw-opt', type=float,
                        help=('The bandwidth you want to use in kernel smoothing. '
                              'HEIG will skip searching the optimal bandwidth. '
                              'For images of any dimension, just specify one number, e.g, 0.5 '
                              'for 3D images'))

# arguments for fpca.py
fpca_parser.add_argument('--image',
                         help='Directory to processed raw images in HDF5 format.')
fpca_parser.add_argument('--sm-image',
                         help='Directory to processed smoothed images in HDF5 format.')
fpca_parser.add_argument('--prop', type=float,
                         help='Proportion of imaging signals to keep, must be a number between 0 and 1.')
fpca_parser.add_argument('--all', action='store_true',
                         help=('Flag for generating all principal components which is min(n_subs, n_voxels), '
                               'which may take longer time and very memory consuming.'))

# arguments for ldmatrix.py
makeld_parser.add_argument('--partition',
                           help=('Genome partition file. '
                                 'The file should be tab or space delimited without header, '
                                 'with the first column being chromosome, '
                                 'the second column being the start position, '
                                 'and the third column being the end position.'
                                 'Each row contains only one LD block.'))
makeld_parser.add_argument('--ld-regu',
                           help=('Regularization for LD matrix and its inverse. '
                                 'Two values should be separated by a comma and between 0 and 1, '
                                 'e.g., `0.85,0.80`'))

# arguments for sumstats.py
sumstats_parser.add_argument('--ldr-gwas',
                             help=('Directory to raw LDR GWAS summary statistics files. '
                                   'Multiple files can be provided using {:}, e.g., `ldr_gwas{1:10}.txt`'))
sumstats_parser.add_argument('--y2-gwas',
                             help='Directory to raw non-imaging GWAS summary statistics file.')
sumstats_parser.add_argument('--n', type=float,
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
                             help=('Genetic effect column, usually refers to beta or odds ratio, '
                                   'should be specified in this format `BETA,0` where '
                                   'BETA is the column name and 0 is the null value. '
                                   'For odds ratio, the null value is 1.'))
sumstats_parser.add_argument('--se-col',
                             help=('Standard error column. For odds ratio, the standard error must be in '
                                   'log(odds ratio) scale.'))
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
sumstats_parser.add_argument('--fast-sumstats', action='store_true',
                             help=('Faster version of processing LDR summary statistics, '
                                   'where only the first LDR is subject to quality checking and SNP pruning.'))

# arguments for voxelgwas.py
voxelgwas_parser.add_argument('--sig-thresh', type=float,
                              help=('p-Value threshold for significance, '
                                    'can be specified in a decimal 0.00000005 '
                                    'or in scientific notation 5e-08.'))

# arguments for vcf2mt.py
annot_vcf_parser.add_argument('--vcf', 
                              help='Direcotory to preprocessed VCF file.')
annot_vcf_parser.add_argument('--favor-db',
                              help='Directory to unzipped FAVOR annotation files.')

# arguments for gwas.py


# arguments for coding.py
wgs_coding_parser.add_argument('--null-model',
                               help='Directory to null model.')
wgs_coding_parser.add_argument('--variant-type',
                               help=("Variant type (case insensitive), "
                                     "must be one of ('variant', 'snv', 'indel')."))
wgs_coding_parser.add_argument('--variant-category',
                               help=("Variant category (case insensitive), "
                                     "must be one or some of ('all', 'plof', 'plof_ds', 'missense', "
                                     "'disruptive_missense', 'synonymous', 'ptv', 'ptv_ds'); "
                                     "where 'all' means all categories; "
                                     "multiple categories should be separated by comma."))
wgs_coding_parser.add_argument('--maf-max', type=float,
                               help='Maximum minor allele frequency for screening SNPs. Default: 0.01')
wgs_coding_parser.add_argument('--mac-thresh', type=int,
                               help='Minimum minor allele count for distinguishing very rare variants. Default: 10.')
wgs_coding_parser.add_argument('--use-annotation-weights', action='store_true',
                               help='If using annotation weights.')


def check_accepted_args(module, args, log):
    accepted_args = {
        'heri_gc': {'out', 'heri_gc', 'ld_inv', 'ld', 'y2_sumstats',
                    'overlap', 'heri_only', 'n_ldrs', 'ldr_sumstats',
                    'bases', 'inner_ldr', 'extract', },
        'kernel_smooth': {'out', 'kernel_smooth', 'keep', 'image_dir', 'image_suffix',
                          'surface_mesh', 'gifti', 'bw_opt'},
        'fpca': {'out', 'fpca', 'image', 'sm_image', 'prop', 'all', 'n_ldrs',
                 'keep', 'covar', 'cat_covar_list'},
        'ld_matrix': {'out', 'ld_matrix', 'partition', 'ld_regu', 'bfile', 'keep',
                      'extract', 'maf_min'},
        'sumstats': {'out', 'sumstats', 'ldr_gwas', 'y2_gwas', 'n', 'n_col',
                     'chr_col', 'pos_col', 'snp_col', 'a1_col',
                     'a2_col', 'effect_col', 'se_col', 'z_col',
                     'p_col', 'maf_col', 'maf_min', 'info_col',
                     'info_min', 'fast_sumstats'},
        'voxel_gwas': {'out', 'voxel_gwas', 'sig_thresh', 'voxel', 'range',
                       'extract', 'ldr_sumstats', 'n_ldrs',
                       'inner_ldr', 'bases'},
        'gwas': {'out', 'gwas', 'ldrs', 'n_ldrs', 'grch37', 'threads', 'geno_mt',
                 'covar', 'cat_covar_list', 'bfile', 'not_save_genotype_data'},
        'annot_vcf': {'annot_vcf', 'out', 'grch37', 'vcf', 'favor_db', 'keep', 'extract'},
        'wgs_null': {'wgs_null', 'out', 'ldrs', 'n_ldrs', 'bases', 'covar',
                     'cat_covar_list', 'keep', 'threads'},
        'wgs_coding': {'wgs_coding', 'out', 'geno_mt', 'null_model', 'variant_type', 
                       'variant_category', 'maf_max', 'maf_min', 'mac_thresh', 
                       'use_annotation_weights', 'n_ldrs', 'keep', 
                       'extract', 'range','voxel', 'not_save_genotype_data'}            
    }

    ignored_args = []
    for k, v in vars(args).items():
        if v is None or not v:
            continue
        elif k not in accepted_args[module]:
            ignored_args.append(k)

    if len(ignored_args) > 0:
        ignored_args = [f"--{arg.replace('_', '-')}" for arg in ignored_args]
        ignored_args_str = ', '.join(ignored_args)
        log.info(f"WARNING: {ignored_args_str} are ignored by --{module.replace('_', '-')}")

    return ignored_args


def split_files(arg):
    files = arg.split(',')
    for file in files:
        if not os.path.exists(file):
            raise ValueError(f"{file} does not exist.")
    return files


def main(args, log):
    dirname = os.path.dirname(args.out)
    if dirname != '' and not os.path.exists(dirname):
        raise ValueError(f'{os.path.dirname(args.out)} does not exist')
    if (args.heri_gc + args.kernel_smooth + args.fpca + args.ld_matrix + args.sumstats + 
        args.voxel_gwas + args.gwas + args.annot_vcf + args.wgs_null + args.wgs_coding != 1):
        raise ValueError(('you must raise one and only one of following flags for doing analysis: '
                          '--heri-gc, --kernel-smooth, --fpca, --ld-matrix, --sumstats, '
                          '--voxel-gwas, --gwas, --annot-vcf, --wgs-null, --wgs-coding'))
    if args.keep is not None:
        args.keep = split_files(args.keep)
    if args.extract is not None:
        args.extract = split_files(args.extract)

    if args.heri_gc:
        check_accepted_args('heri_gc', args, log)
        import heig.herigc as herigc
        herigc.run(args, log)
    elif args.kernel_smooth:
        check_accepted_args('kernel_smooth', args, log)
        import heig.ksm as ksm
        ksm.run(args, log)
    elif args.fpca:
        check_accepted_args('fpca', args, log)
        import heig.fpca as fpca
        fpca.run(args, log)
    elif args.ld_matrix:
        check_accepted_args('ld_matrix', args, log)
        import heig.ldmatrix as ldmatrix
        ldmatrix.run(args, log)
    elif args.sumstats:
        check_accepted_args('sumstats', args, log)
        import heig.sumstats as sumstats
        sumstats.run(args, log)
    elif args.voxel_gwas:
        check_accepted_args('voxel_gwas', args, log)
        import heig.voxelgwas as voxelgwas
        voxelgwas.run(args, log)
    elif args.gwas:
        check_accepted_args('gwas', args, log)
        import heig.wgs.gwas as gwas
        gwas.run(args, log)
    elif args.annot_vcf:
        check_accepted_args('annot_vcf', args, log)
        import heig.wgs.vcf2mt as vcf2mt
        vcf2mt.run(args, log)
    elif args.wgs_null:
        check_accepted_args('wgs_null', args, log)
        import heig.wgs.null as null
        null.run(args, log)
    elif args.wgs_coding:
        check_accepted_args('wgs_coding', args, log)
        import heig.wgs.coding as coding
        coding.run(args, log)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.out is None:
        args.out = 'heig'

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "heig.py \\\n"
        options = ['--'+x.replace('_', '-')+' ' +
                   str(opts[x]) + ' \\' for x in non_defaults]
        header += '\n'.join(options).replace(' True', '').replace(' False', '')
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
