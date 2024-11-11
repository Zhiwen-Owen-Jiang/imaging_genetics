import os
import time
import argparse
import traceback
import numexpr
import numpy as np
import heig.input.dataset as ds
from heig.utils import GetLogger, sec_to_str


# os.environ['NUMEXPR_MAX_THREADS'] = '8'
# numexpr.set_num_threads(int(os.environ['NUMEXPR_MAX_THREADS']))


VERSION = "1.1.0"
MASTHEAD = (
    "******************************************************************************\n"
)
MASTHEAD += "* Highly Efficient Imaging Genetics (HEIG)\n"
MASTHEAD += f"* Version {VERSION}\n"
MASTHEAD += f"* Zhiwen Jiang and Hongtu Zhu\n"
MASTHEAD += (
    f"* Department of Biostatistics, University of North Carolina at Chapel Hill\n"
)
MASTHEAD += f"* GNU General Public License v3\n"
MASTHEAD += f"* Correspondence: owenjf@live.unc.edu, zhiwenowenjiang@gmail.com\n"
MASTHEAD += (
    "******************************************************************************\n"
)


parser = argparse.ArgumentParser(
    description=f"\n Highly Efficient Imaging Genetics (HEIG) v{VERSION}"
)

common_parser = parser.add_argument_group(title="Common arguments")
herigc_parser = parser.add_argument_group(
    title="Arguments specific to heritability and (cross-trait) genetic correlation analysis"
)
image_parser = parser.add_argument_group(title="Arguments specific to reading images")
fpca_parser = parser.add_argument_group(title="Arguments specific to functional PCA")
ldr_parser = parser.add_argument_group(title="Arguments specific to constructing LDRs")
makeld_parser = parser.add_argument_group(
    title="Arguments specific to making an LD matrix and its inverse"
)
sumstats_parser = parser.add_argument_group(
    title="Arguments specific to organizing and preprocessing GWAS summary statistics"
)
voxelgwas_parser = parser.add_argument_group(
    title="Arguments specific to recovering voxel-level GWAS results"
)
gwas_parser = parser.add_argument_group(
    title="Arguments specific to doing genome-wide association analysis"
)
annot_parser = parser.add_argument_group(
    title="Arguments specific to annotating genotype files"
)
wgs_null_parser = parser.add_argument_group(
    title="Arguments specific to the null model of whole genome sequencing analysis"
)
wgs_coding_parser = parser.add_argument_group(
    title="Arguments specific to whole genome sequencing analysis for coding variants"
)
wgs_sliding_window_parser = parser.add_argument_group(
    title="Arguments specific to whole genome sequencing analysis using sliding windows"
)
relatedness_parser = parser.add_argument_group(
    title="Arguments specific to removing genetic relatedness in LDRs"
)
wgs_general_annot_parser = parser.add_argument_group(
    title="Arguments specific to whole genome sequencing analysis using general annotations"
)
wgs_sumstats_parser = parser.add_argument_group(
    title="Arguments specific to generating whole genome sequencing analysis summary statistics"
)

# module arguments
herigc_parser.add_argument(
    "--heri-gc",
    action="store_true",
    help="Heritability and (cross-trait) genetic correlation analysis.",
)
image_parser.add_argument("--read-image", action="store_true", help="Reading images.")
fpca_parser.add_argument("--fpca", action="store_true", help="Functional PCA.")
ldr_parser.add_argument("--make-ldr", action="store_true", help="Constructing LDRs.")
makeld_parser.add_argument(
    "--ld-matrix", action="store_true", help="Making an LD matrix and its inverse."
)
sumstats_parser.add_argument(
    "--sumstats",
    action="store_true",
    help="Organizing and preprocessing GWAS summary statistics.",
)
voxelgwas_parser.add_argument(
    "--voxel-gwas", action="store_true", help="Recovering voxel-level GWAS results."
)
gwas_parser.add_argument(
    "--gwas", action="store_true", help="Genome-wide association analysis."
)
annot_parser.add_argument(
    "--annot", action="store_true", help="Annotating genotype files."
)
wgs_null_parser.add_argument(
    "--wgs-null",
    action="store_true",
    help="Fitting the null model of whole genome sequencing analysis.",
)
wgs_coding_parser.add_argument(
    "--wgs-coding",
    action="store_true",
    help="Whole genome sequencing analysis for coding variants.",
)
wgs_sliding_window_parser.add_argument(
    "--wgs-sliding-window",
    action="store_true",
    help="Whole genome sequencing analysis using sliding windows.",
)
relatedness_parser.add_argument(
    "--relatedness", action="store_true", help="Removing genetic relatedness in LDRs."
)
wgs_general_annot_parser.add_argument(
    "--wgs", action="store_true", 
    help="Whole genome sequencing analysis using general annotations."
)
wgs_sumstats_parser.add_argument(
    "--wgs-sumstats", action="store_true", 
    help="Generating whole genome sequencing analysis summary statistics."
)

# common arguments
common_parser.add_argument("--out", help="Prefix of output.")
common_parser.add_argument(
    "--image",
    help=(
        "Directory to processed raw images in HDF5 format. "
        "Supported modules: --fpca, --make-ldr."
    ),
)
common_parser.add_argument(
    "--n-ldrs",
    type=int,
    help=(
        "Number of LDRs. Supported modules: "
        "--make-ldr, --fpca, --heri-gc, --voxel-gwas, --wgs-null, "
        "--wgs-coding, --wgs-sliding-window, --relatedness."
    ),
)
common_parser.add_argument(
    "--ldr-sumstats",
    help=(
        "Prefix of preprocessed LDR GWAS summary statistics. "
        "Supported modules: --heri-gc, --voxel-gwas."
    ),
)
common_parser.add_argument(
    "--bases",
    help=(
        "Directory to functional bases. Supported modules: "
        "--make-ldr, --heri-gc, --voxel-gwas, --wgs-null."
    ),
)
common_parser.add_argument(
    "--ldr-cov",
    help=(
        "Directory to variance-covariance marix of LDRs. "
        "Supported modules: --heri-gc, --voxel-gwas."
    ),
)
common_parser.add_argument(
    "--keep",
    help=(
        "Individual file(s). Multiple files are separated by comma. "
        "Each file should be tab or space delimited, "
        "with the first column being FID and the second column being IID. "
        "Other columns will be ignored. "
        "Each row contains only one subject. "
        "Supported modules: --read-image, --fpca, --make-ldr, --ld-matrix, "
        "--wgs-null, --wgs-coding, --wgs-sliding-window, --relatedness."
    ),
)
common_parser.add_argument(
    "--extract",
    help=(
        "SNP file(s). Multiple files are separated by comma. "
        "Each file should be tab or space delimited, "
        "with the first column being rsID. "
        "Other columns will be ignored. "
        "Each row contains only one SNP. "
        "Supported modules: --heri-gc, --ld-matrix, --voxel-gwas, "
        "--wgs-coding, --wgs-sliding-window, --relatedness."
    ),
)
common_parser.add_argument(
    "--maf-min",
    type=float,
    help=(
        "Minimum minor allele frequency for screening SNPs. "
        "Supported modules: --ld-matrix, --sumstats, "
        "--wgs-coding, --wgs-sliding-window, --relatedness."
    ),
)
common_parser.add_argument(
    "--covar",
    help=(
        "Directory to covariate file. "
        "The file should be tab or space delimited, with each row only one subject. "
        "Supported modules: --make-ldr, --gwas, --wgs-null, --relatedness."
    ),
)
common_parser.add_argument(
    "--cat-covar-list",
    help=(
        "List of categorical covariates to include in the analysis. "
        "Multiple covariates are separated by comma. "
        "Supported modules: --make-ldr, --gwas, --wgs-null, --relatedness."
    ),
)
common_parser.add_argument(
    "--bfile",
    help=(
        "Prefix of PLINK bfile triplets. "
        "When estimating LD matrix and its inverse, two prefices should be provided "
        "and seperated by a comma, e.g., `prefix1,prefix2`. "
        "When doing GWAS, only one prefix is allowed. "
        "Supported modules: --ld-matrix, --gwas, --relatedness."
    ),
)
common_parser.add_argument(
    "--range",
    help=(
        "A segment of chromosome, e.g. `3:1000000,3:2000000`, "
        "from chromosome 3 bp 1000000 to chromosome 3 bp 2000000. "
        "Cross-chromosome is not allowed. And the end position must "
        "be greater than the start position. "
        "Supported modules: --voxel-gwas, --wgs-coding, "
        "--wgs-sliding-window."
    ),
)
common_parser.add_argument(
    "--voxel",
    help=(
        "one-based index of voxel or a file containing voxels. "
        "Supported modules: --voxel-gwas, --wgs-coding, "
        "--wgs-sliding-window."
    ),
)
common_parser.add_argument(
    "--ldrs",
    help=(
        "Directory to LDR file. "
        "Supported modules: --gwas, --wgs-null, --relatedness."
    ),
)
common_parser.add_argument(
    "--geno-mt",
    help=(
        "Directory to genotype MatrixTable. "
        "Supported modules: --gwas, --wgs-coding, --wgs-sliding-window, "
        "--relatedness."
    ),
)
common_parser.add_argument(
    "--grch37",
    action="store_true",
    help=(
        "Using reference genome GRCh37. Otherwise using GRCh38. "
        "Supported modules: --gwas, --annot, --wgs-sliding-window, "
        "--relatedness"
    ),
)
common_parser.add_argument(
    "--not-save-genotype-data",
    action="store_true",
    help=(
        "Do not save preprocessed genotype data. "
        "Supported modules: --gwas, --wgs-coding, --wgs-sliding-window, "
        "--relatedness"
    ),
)  # may remove it
common_parser.add_argument(
    "--partition",
    help=(
        "Genome partition file. "
        "The file should be tab or space delimited without header, "
        "with the first column being chromosome, "
        "the second column being the start position, "
        "and the third column being the end position."
        "Each row contains only one LD block. "
        "Supported modules: --ld-matrix, --relatedness."
    ),
)
common_parser.add_argument(
    "--threads",
    type=int,
    help=(
        "number of threads. "
        "Supported modules: --read-image, --sumstats, --fpca, "
        "--voxel-gwas, --heri-gc, --make-ldr."
    ),
),
common_parser.add_argument(
    "--spark-conf",
    help=(
        "Spark configuration file. "
        "Supported modules: --relatedness, --gwas, --wgs-coding, "
        "--wgs-noncoding, --wgs-sliding-window, --annot."
    ),
),
common_parser.add_argument(
    "--loco-preds",
    help=(
        "Leave-one-chromosome-out prediction file. "
        "Supported modules: --gwas, --coding, "
        "--noncoding, --slidingwindow."
    ),
),
common_parser.add_argument(
    "--mac-thresh",
    type=int,
    help=(
        "A minor allele count threshold. "
        "Variants with a MAC less than the threshold "
        "will be marked as a rare variants in WGS ACAT-V analysis. "
        "Default is 10. "
        "Supported modules: --annot, --coding. " # TODO: add to more modules
    ),
),
common_parser.add_argument(
    "--call-rate",
    type=float,
    help=(
        "A genotype call rate threshold, equivalent to 1 - missing rate. "
        "Variants with a call rate less than the threshold "
        "will be removed."
        "Supported modules: --annot, --coding. " # TODO: add to more modules
    ),
),
common_parser.add_argument(
    "--hwe",
    type=float,
    help=(
        "A HWE p-value threshold. "
        "Variants with a HWE p-value less than the threshold "
        "will be removed."
        "Supported modules: --annot, --coding. " # TODO: add to more modules
    ),
)


# arguments for herigc.py
herigc_parser.add_argument(
    "--ld-inv",
    help=(
        "Prefix of inverse LD matrix. Multiple matrices can be specified using {:}, "
        "e.g., `ld_inv_chr{1:22}_unrel`."
    ),
)
herigc_parser.add_argument(
    "--ld",
    help=(
        "Prefix of LD matrix. Multiple matrices can be specified using {:}, "
        "e.g., `ld_chr{1:22}_unrel`."
    ),
)
herigc_parser.add_argument(
    "--y2-sumstats",
    help="Prefix of preprocessed GWAS summary statistics of non-imaging traits.",
)
herigc_parser.add_argument(
    "--overlap",
    action="store_true",
    help=(
        "Flag for indicating sample overlap between LDR summary statistics "
        "and non-imaging summary statistics. Only effective if --y2-sumstats is specified."
    ),
)
herigc_parser.add_argument(
    "--heri-only",
    action="store_true",
    help=(
        "Flag for only computing voxelwise heritability "
        "and skipping voxelwise genetic correlation within images."
    ),
)

# arguments for image.py
image_parser.add_argument(
    "--image-txt",
    help=(
        "Directory to images in txt format. "
        "The file should be tab or space delimited, with each row only one subject."
    ),
)
image_parser.add_argument(
    "--coord-txt",
    help=(
        "Directory to images in txt format. "
        "The file should be tab or space delimited, with each row only one voxel (vertex)."
    ),
)
image_parser.add_argument(
    "--image-dir",
    help=(
        "Directory to images. All images in the directory with matched suffix "
        "(see --image-suffix) will be loaded. "
        "Multiple directories can be provided and separated by comma. "
        "--keep can be used to load a subset of images (see --keep). "
        "The supported formats include NIFTI and CIFTI images "
        "and FreeSurfer morphometry data file."
    ),
)
image_parser.add_argument(
    "--image-suffix",
    help=(
        "Suffix of images. HEIG requires the name of each image in the format <ID><suffix>, "
        "e.g., `1000001_masked_FAskel.nii.gz`, where `1000001` is the ID "
        "and `_masked_FAskel.nii.gz` is the suffix. "
        "HEIG will collect ID for each image. "
        "Multiple suffixes can be specified and separated by comma "
        "and the number of directories must match the number of suffices."
    ),
)
image_parser.add_argument(
    "--coord-dir",
    help=(
        "Directory to mask or complementary image for coordinates. "
        "It should be a NIFTI file (nii.gz) for NIFTI images; "
        "a GIFTI file (gii) for CIFTI2 surface data; "
        "a FreeSurfer surface mesh file (.pial) for FreeSurfer morphometry data."
    ),
)

# arguments for fpca.py
fpca_parser.add_argument(
    "--all-pc",
    action="store_true",
    help=(
        "Flag for generating all principal components which is min(n_subs, n_voxels), "
        "which may take longer time and very memory consuming."
    ),
)
fpca_parser.add_argument(
    "--bw-opt",
    type=float,
    help=(
        "The bandwidth you want to use in kernel smoothing. "
        "HEIG will skip searching the optimal bandwidth. "
        "For images of any dimension, just specify one number, e.g, 0.5 "
        "for 3D images"
    ),
)
fpca_parser.add_argument(
    "--skip-smoothing",
    action='store_true',
    help=(
        "Skip kernel smoothing. "
    ),
)

# arguments for ldmatrix.py
makeld_parser.add_argument(
    "--ld-regu",
    help=(
        "Regularization for LD matrix and its inverse. "
        "Two values should be separated by a comma and between 0 and 1, "
        "e.g., `0.85,0.80`"
    ),
)

# arguments for sumstats.py
sumstats_parser.add_argument(
    "--ldr-gwas",
    help=(
        "Directory to raw LDR GWAS summary statistics files. "
        "Multiple files can be provided using {:}, e.g., `ldr_gwas{1:10}.txt`."
    ),
)
sumstats_parser.add_argument(
    "--ldr-gwas-heig",
    help=(
        "Directory to raw LDR GWAS summary statistics files produced by --gwas. "
        "Multiple files can be provided using {:}, e.g., `ldr_gwas{1:10}.txt.bgz`. "
        "One file may contain multiple LDRs. These files must be in order."
    ),
)
sumstats_parser.add_argument(
    "--y2-gwas", help="Directory to raw non-imaging GWAS summary statistics file."
)
sumstats_parser.add_argument("--n", type=float, help="Sample size. A positive number.")
sumstats_parser.add_argument("--n-col", help="Sample size column.")
sumstats_parser.add_argument("--chr-col", help="Chromosome column.")
sumstats_parser.add_argument("--pos-col", help="Position column.")
sumstats_parser.add_argument("--snp-col", help="SNP column.")
sumstats_parser.add_argument("--a1-col", help="A1 column. The effective allele.")
sumstats_parser.add_argument("--a2-col", help="A2 column. The non-effective allele.")
sumstats_parser.add_argument(
    "--effect-col",
    help=(
        "Genetic effect column, usually refers to beta or odds ratio, "
        "should be specified in this format `BETA,0` where "
        "BETA is the column name and 0 is the null value. "
        "For odds ratio, the null value is 1."
    ),
)
sumstats_parser.add_argument(
    "--se-col",
    help=(
        "Standard error column. For odds ratio, the standard error must be in "
        "log(odds ratio) scale."
    ),
)
sumstats_parser.add_argument("--z-col", help="Z score column.")
sumstats_parser.add_argument("--p-col", help="p-Value column.")
sumstats_parser.add_argument("--maf-col", help="Minor allele frequency column.")
sumstats_parser.add_argument("--info-col", help="INFO score column.")
sumstats_parser.add_argument(
    "--info-min", type=float, help="Minimum INFO score for screening SNPs."
)

# arguments for voxelgwas.py
voxelgwas_parser.add_argument(
    "--sig-thresh",
    type=float,
    help=(
        "p-Value threshold for significance, "
        "can be specified in a decimal 0.00000005 "
        "or in scientific notation 5e-08."
    ),
)

# arguments for vcf2mt.py
annot_parser.add_argument("--vcf", help="Direcotory to preprocessed VCF file.")
annot_parser.add_argument(
    "--favor-annot", 
    help=(
        "Directory to unzipped FAVOR annotation files. "
        "For multiple files, using * to match any string of characters. "
        "E.g., favor_db/chr*.csv"
    ),
)
annot_parser.add_argument(
    "--general-annot", 
    help=(
        "Directory to general annotation files. "
        "Following the file format of FAVOR database, "
        "columns `chromosome`, `position`, `ref_vcf`, and `alt_vcf` are required. "
        "Columns must be separated by comma, and missing values are not allowed. "
        "Use double quote marks `\"`"
        "For multiple files, using * to match any string of characters. "
        "E.g., chr*.csv"
    ),
)

# arguments for coding.py
wgs_coding_parser.add_argument("--null-model", help="Directory to null model.")
wgs_coding_parser.add_argument(
    "--variant-type",
    help=(
        "Variant type (case insensitive), "
        "must be one of ('variant', 'snv', 'indel')."
    ),
)
wgs_coding_parser.add_argument(
    "--variant-category",
    help=(
        "Variant category (case insensitive), "
        "must be one or some of ('all', 'plof', 'plof_ds', 'missense', "
        "'disruptive_missense', 'synonymous', 'ptv', 'ptv_ds'); "
        "where 'all' means all categories; "
        "multiple categories should be separated by comma."
    ),
)
wgs_coding_parser.add_argument(
    "--maf-max",
    type=float,
    help="Maximum minor allele frequency for screening SNPs. Default: 0.01",
)
wgs_coding_parser.add_argument(
    "--use-annotation-weights", action="store_true", help="If using annotation weights."
)

# arguments for slidingwindow.py
wgs_sliding_window_parser.add_argument(
    "--window-length", type=int, help="Fix window length. Default: 2000."
)

# arguments for general_annot.py
wgs_general_annot_parser.add_argument(
    "--annot-name", help="Annotation to use. Multiple annotations should be separated by comma."
)

# arguments for relatedness.py
relatedness_parser.add_argument(
    "--bsize", type=int, help="Block size of genotype blocks. Default: 5000."
)

# arguments for gwas.py
gwas_parser.add_argument(
    "--ldr-col", help="One-based LDR indices. E.g., `3,4,5,6` and `3:6`, must be consecutive"
)


def check_accepted_args(module, args, log):
    """
    Checking if the provided arguments are accepted by the module

    """
    accepted_args = {
        "heri_gc": {
            "out",
            "heri_gc",
            "ld_inv",
            "ld",
            "y2_sumstats",
            "overlap",
            "heri_only",
            "n_ldrs",
            "ldr_sumstats",
            "bases",
            "ldr_cov",
            "extract",
            "threads",
        },
        "read_image": {
            "out",
            "read_image",
            "keep",
            "image_txt",
            "coord_txt",
            "image_dir",
            "image_suffix",
            "coord_dir",
            "threads",
        },
        "fpca": {
            "out",
            "fpca",
            "image",
            "all_pc",
            "n_ldrs",
            "keep",
            "bw_opt",
            "skip_smoothing",
            "threads",
        },
        "make_ldr": {
            "out",
            "make_ldr",
            "image",
            "bases",
            "n_ldrs",
            "covar",
            "cat_covar_list",
            "keep",
            "threads",
        },
        "ld_matrix": {
            "out",
            "ld_matrix",
            "partition",
            "ld_regu",
            "bfile",
            "keep",
            "extract",
            "maf_min",
        },
        "sumstats": {
            "out",
            "sumstats",
            "ldr_gwas",
            "y2_gwas",
            "ldr_gwas_heig",
            "n",
            "n_col",
            "chr_col",
            "pos_col",
            "snp_col",
            "a1_col",
            "a2_col",
            "effect_col",
            "se_col",
            "z_col",
            "p_col",
            "maf_col",
            "maf_min",
            "info_col",
            "info_min",
            "threads",
        },
        "voxel_gwas": {
            "out",
            "voxel_gwas",
            "sig_thresh",
            "voxel",
            "range",
            "extract",
            "ldr_sumstats",
            "n_ldrs",
            "ldr_cov",
            "bases",
            "threads",
        },
        "gwas": {
            "out",
            "gwas",
            "ldr_col",
            "ldrs",
            "n_ldrs",
            "grch37",
            "threads",
            "geno_mt",
            "covar",
            "cat_covar_list",
            "bfile",
            "loco_preds",
            "spark_conf",
            "not_save_genotype_data",
        },
        "annot": {
            "annot",
            "out",
            "grch37",
            "vcf",
            "bfile",
            "favor_annot",
            "general_annot",
            "keep",
            "extract",
            "spark_conf",
            "hwe",
            "call_rate",
            "maf_min",
            "maf_max"
        },
        "wgs_null": {
            "wgs_null",
            "out",
            "ldrs",
            "n_ldrs",
            "bases",
            "covar",
            "cat_covar_list",
            "keep",
            "threads",
        },
        "wgs_coding": {
            "wgs_coding",
            "out",
            "geno_mt",
            "null_model",
            "variant_type",
            "variant_category",
            "maf_max",
            "maf_min",
            "mac_thresh",
            "use_annotation_weights",
            "grch37",
            "loco_preds",
            "n_ldrs",
            "keep",
            "extract",
            "range",
            "voxel",
            "not_save_genotype_data",
            "spark_conf",
            "hwe",
            "call_rate",
            "mac_thresh"
        },
        "wgs": {
            "wgs",
            "out",
            "geno_mt",
            "null_model",
            "variant_type",
            "annot_name",
            "maf_max",
            "maf_min",
            "mac_thresh",
            "n_ldrs",
            "keep",
            "extract",
            "grch37",
            "loco_preds",
            "range",
            "voxel",
            "not_save_genotype_data",
            "spark_conf",
            "hwe",
            "call_rate",
            "mac_thresh"
        },
        "wgs_sliding_window": {
            "wgs_sliding_window",
            "out",
            "geno_mt",
            "null_model",
            "variant_type",
            "window_length",
            "maf_max",
            "maf_min",
            "mac_thresh",
            "use_annotation_weights",
            "n_ldrs",
            "keep",
            "extract",
            "range",
            "voxel",
            "not_save_genotype_data",
            "spark_conf"
        },
        "relatedness": {
            "relatedness",
            "out",
            "ldrs",
            "covar",
            "cat_covar_list",
            "bfile",
            "partition",
            "maf_min",
            "n_ldrs",
            "grch37",
            "geno_mt",
            "not_save_genotype_data",
            "bsize",
            "spark_conf",
            "threads"
        },  # more arguments to add
        "wgs_sumstats": {
            "wgs_sumstats",
            "out",
            "geno_mt",
            "bfile",
            "vcf",
            "null_model",
            "variant_type",
            "maf_max",
            "maf_min",
            "mac_thresh",
            "grch37",
            "loco_preds",
            "n_ldrs",
            "keep",
            "extract",
            "range",
            "voxel",
            "not_save_genotype_data",
            "spark_conf",
            "hwe",
            "call_rate",
            "mac_thresh"
        },
    }

    ignored_args = []
    for k, v in vars(args).items():
        if v is None or not v:
            continue
        elif k not in accepted_args[module]:
            ignored_args.append(k)
            setattr(args, k, None)

    if len(ignored_args) > 0:
        ignored_args = [f"--{arg.replace('_', '-')}" for arg in ignored_args]
        ignored_args_str = ", ".join(ignored_args)
        log.info(
            f"WARNING: {ignored_args_str} ignored by --{module.replace('_', '-')}."
        )


def split_files(arg):
    files = arg.split(",")
    for file in files:
        ds.check_existence(file)
    return files


def process_args(args, log):
    """
    Checking file existence and processing arguments

    """
    ds.check_existence(args.image)
    ds.check_existence(args.ldr_sumstats, ".snpinfo")
    ds.check_existence(args.ldr_sumstats, ".sumstats")
    ds.check_existence(args.bases)
    ds.check_existence(args.ldr_cov)
    ds.check_existence(args.covar)
    ds.check_existence(args.partition)
    ds.check_existence(args.ldrs)
    ds.check_existence(args.geno_mt)
    ds.check_existence(args.null_model)

    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError("--n-ldrs must be greater than 0")

    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be greater than 0")
    else:
        args.threads = 1
    log.info(f"Using {args.threads} thread(s) in analysis.")

    if args.keep is not None:
        args.keep = split_files(args.keep)
        args.keep = ds.read_keep(args.keep)
        log.info(f"{len(args.keep)} subjects in --keep.")

    if args.extract is not None:
        args.extract = split_files(args.extract)
        args.extract = ds.read_extract(args.extract)
        log.info(f"{len(args.extract)} SNPs in --extract.")
        
    if args.bfile is not None:
        for suffix in [".bed", ".fam", ".bim"]:
            ds.check_existence(args.bfile, suffix)

    if args.voxel is not None:
        try:
            args.voxel = np.array(
                [int(voxel) - 1 for voxel in ds.parse_input(args.voxel)]
            )
        except ValueError:
            ds.check_existence(args.voxel)
            args.voxel = ds.read_voxel(args.voxel)
        if np.min(args.voxel) <= -1:
            raise ValueError("voxel index must be one-based")

    if args.maf_min is not None:
        if args.maf_min > 0.5 or args.maf_min < 0:
            raise ValueError("--maf-min must be greater than 0 and less than 0.5")
    # else:
    #     args.maf_min = 0 # >


def main(args, log):
    dirname = os.path.dirname(args.out)
    if dirname != "" and not os.path.exists(dirname):
        raise ValueError(f"{os.path.dirname(args.out)} does not exist")
    if (
        args.heri_gc
        + args.read_image
        + args.fpca
        + args.make_ldr
        + args.ld_matrix
        + args.sumstats
        + args.voxel_gwas
        + args.gwas
        + args.annot
        + args.wgs_null
        + args.wgs_coding
        + args.wgs_sliding_window
        + args.wgs
        + args.relatedness
        + args.wgs_sumstats
        != 1
    ):
        raise ValueError(
            (
                "you must raise one and only one of following flags for doing analysis: "
                "--heri-gc, --read-image, --fpca, --make-ldr, --ld-matrix, --sumstats, "
                "--voxel-gwas, --gwas, --annot, --wgs-null, --wgs-coding, "
                "--wgs-sliding-window, --wgs, --relatedness, --wgs-sumstats"
            )
        )

    if args.heri_gc:
        check_accepted_args("heri_gc", args, log)
        import heig.herigc as module
    elif args.read_image:
        check_accepted_args("read_image", args, log)
        import heig.image as module
    elif args.fpca:
        check_accepted_args("fpca", args, log)
        import heig.fpca as module
    elif args.make_ldr:
        check_accepted_args("make_ldr", args, log)
        import heig.ldr as module
    elif args.ld_matrix:
        check_accepted_args("ld_matrix", args, log)
        import heig.ldmatrix as module
    elif args.sumstats:
        check_accepted_args("sumstats", args, log)
        import heig.sumstats as module
    elif args.voxel_gwas:
        check_accepted_args("voxel_gwas", args, log)
        import heig.voxelgwas as module
    elif args.gwas:
        check_accepted_args('gwas', args, log)
        import heig.wgs.gwas as module
    elif args.annot:
        check_accepted_args('annot', args, log)
        import heig.wgs.vcf2mt as module
    elif args.wgs_null:
        check_accepted_args('wgs_null', args, log)
        import heig.wgs.null as module
    elif args.wgs_coding:
        check_accepted_args('wgs_coding', args, log)
        import heig.wgs.coding as module
    elif args.wgs_sliding_window:
        check_accepted_args('wgs_sliding_window', args, log)
        import heig.wgs.slidingwindow as module
    elif args.wgs:
        check_accepted_args('wgs', args, log)
        import heig.wgs.general_annot as module
    elif args.relatedness:
        check_accepted_args('relatedness', args, log)
        import heig.wgs.relatedness as module
    elif args.wgs_sumstats:
        check_accepted_args('wgs_sumstats', args, log)
        import heig.wgs2.wgs as module

    process_args(args, log)
    module.run(args, log)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "heig"

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(""))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "heig.py \\\n"
        options = [
            "--" + x.replace("_", "-") + " " + str(opts[x]) + " \\"
            for x in non_defaults
        ]
        header += "\n".join(options).replace(" True", "").replace(" False", "")
        header = header + "\n"
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"\nAnalysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")
