## 0. Getting started
In this tutorial, we will go through seven analysis modules in **HEIG** (v.1.2.0), including reading images, functional PCA, constructing low‐dimensional representations (LDRs), conducting LDR GWAS, processing summary statistics, voxelwise GWAS reconstruction, heritability and (cross-trait) genetic correlation analysis. For beginners, we recommend reading these modules in the above sequence. Note this tutorial may not be compatible with earlier versions.

To replicate the analyses showed in the tutorial, you can download the example data at [Zenodo](https://zenodo.org/records/13770930). The total file size is 1.1 GB. After downloading the example data, unzip it and navigate into the folder. Make sure your current directory is `example`. Let's get started.


## 1. Reading images
To read `NIFTI` images saved at `input/images/` with naming convention `<ID>_example_image.nii.gz` and the corresponding coordinate information in a `NIFTI` image `input/images/s1000_example_image.nii.gz`:
```
heig.py \
--read-image \
--out output/images/example \
--threads 4 \
--image-dir input/images1,input/images2 \
--image-suffix _example_image.nii.gz,_example_image.nii.gz \
--coord-dir input/images1/s1000_example_image.nii.gz \
```
`--read-image` (required): the flag for reading images.

`--out` (required): the prefix of output. 

`--threads` (optional): the number of threads in parallel.

`--image-dir` + `--image-suffix` + `--coord-dir` is a combination to specify image directory(s), image suffix(s), and coordinates. Alternatively, `--image-txt` + `--coord-txt` specifies images and coordinates in text format.

#### Notes
**HEIG** supports images in `NIFTI`, `CIFTI2`, and FreeSurfer morphometry data format. Before loading into **HEIG**, all images should be appropriately registered/projected to the same template. Images can be placed under one or more directories. Separate multiple directories by comma, such as `input/images1,input/images2`. The naming convention of image file is `<ID><suffix>`. For example, images in this tutorial were named as `sxxxx_example_image.nii.gz`, where `sxxxx` (e.g., s1001) is the ID and `_example_image.nii.gz` is the suffix. If images are from different directories, the same number of suffices must be provided and separated by comma, such as `_example_image.nii.gz,_example_image.nii.gz`.

`--coord-dir` is to specify the coordinate file. For `NIFTI` images, it should also be a `NIFTI` image; for `CIFTI2` images, it should be a `GIFTI` image; for FreeSurfer morphometry data, it should be a FreeSurfer surface mesh file. Only one coordinate file can be provided even if we have images from multiple directories.

**HEIG** also supports non-imaging phenotypes in tabular data. Using `--image-txt` and `--coord-txt` to read the dataset and coordinates, respectively. The coordinate file can be an arbitrary text file with the number of rows the same as the number of phenotypes.

#### Output
- `output/images/example_images.h5`: images, coordinates, and subject IDs saved in HDF5 format.

#### Further reading
Refer to Data Management for merging multiple image files, and keeping and excluding subjects from image files.


## 2. Functional PCA
To conduct Functional PCA analysis computing all principal components (PCs):
```
heig.py \
--fpca \
--out output/fpca/example \
--image output/images/example_images.h5 \
--threads 4 \
--all-pc \
```
`--fpca` (required): the flag for functional PCA.

`--out` (required): the prefix of output. 

`--image` (required): the image file.

`--threads` (optional): the number of threads in parallel.

`--all-pc` (optional): a flag of computing all PCs.

#### Notes
Additional flags include `--bw-opt`, `--n-ldrs` and `--skip-smoothing`. 

`--bw-opt` tells HEIG the optimal bandwidth so that HEIG can skip the bandwidth selection procedure. For images with any dimension, just specify a single positive number (e.g. `--bw-opt 0.5`), since we assume that the bandwidths for all dimensions are the same. 

`--n-ldrs` specifies the specific number of PCs to compute, which is useful when computing only top PCs. Uncomputed eigenvalues will be imputed using a `B-spline` model of degree 1.

`--skip-smoothing` is to skip kernel smoothing, then functional PCA reduces to PCA. It is useful when analyzing non-imaging phenotypes.

#### Output
- `output/fpca/example_bases_top594.npy`: top functional bases (eigenvectors) from functional PCA.
- `output/fpca/example_eigenvalues.npy`: all eigenvalues.
- `output/fpca/example_ldrs_prop_var.txt`: a text file saving the number of LDRs for preserving varying proportions of image variance.


## 3. Constructing LDRs
To construct 23 LDRs and adjust for covariates to calculate covariate-effect-removed variance-covariance matrix of LDRs:
```
heig.py \
--make-ldr \
--out output/ldr/example \
--image output/images/example_images.h5 \
--n-ldrs 23 \
--bases output/fpca/example_bases_top594.npy \
--covar input/misc/covar.txt \
--cat-covar-list sex,imaging_site \
--threads 4 \
```
`--make-ldr` (required): the flag for constructing LDRs.

`--out` (required): the prefix of output.

`--image` (required): the image file.

`--n-ldrs` (optional): the number of LDRs. Otherwise, analysis will be performed for all LDRs.

`--bases` (required): the functional bases.

`--covar` (required): the covariate file that will be used in GWAS. 

`--cat-covar-list` (optional): categorical covariates each separated by comma.

`--threads` (optional): the number of threads in parallel.

#### Notes
The variance-covariance matrix of LDRs is used to recover standard error estimates for voxel-level genetic effects as well as to recover voxel variance in voxel heritability analysis. The caveat is that it is calculated before doing LDR GWAS. It is likely that only a subset of samples have genetic data and are actually used in GWAS, resulting in sample inconsistency between genetic effect estimates and standard error estimates in voxel-level summary statistics reconstruction. We encourage users to use `--keep` and `--remove` here to match subjects that will be used in LDR GWAS. Additionally, exclude variants with a high missing rate (e.g. 0.1) from the genotype data before doing GWAS.

#### Output
- `output/ldr/example_ldr_top23.txt`: the top 23 LDRs.
- `output/ldr/example_ldr_cov_top23.npy`: the variance-covariance matrix of the top 23 covariate-effect-removed LDRs.


## 4. Conducting LDR GWAS
To conduct LDR GWAS without considering relatedness (we used the genotype data from 1000 Genomes):
```
heig.py \
--gwas \
--out output/gwas/ldr_1kg \
--n-ldrs 5 \
--covar input/misc/covar_1kg.txt \
--cat-covar-list sex,imaging_site \
--ldrs input/misc/ldr_1kg_top10.txt \
--geno-mt input/misc/1kg_1ksnps.mt \
--grch37 \
--not-save-genotype-data \
--spark-conf input/misc/spark_config_small_mem.json \
```
`--gwas` (required): the flag for conducting LDR GWAS.

`--out` (required): the prefix of output.

`--n-ldrs` (optional): the number of top LDRs. Otherwise, analysis will be performed for all LDRs.

`--covar` (required): the covariate file that used in constructing LDRs. 

`--cat-covar-list` (optional): categorical covariates each separated by comma.

`--ldrs` (required): the LDR file.

`--geno-mt` (optional): a `hail.MatrixTable` of genotype data.

`--grch37` (optional): a flag indicating the reference genome is GRCh37. Default is `False`.

`--not-save-genotype-data` (optional): a flag of not saving genotype data. It is useful to save interim genotype data when you want to do complex QC or subsetting genotype data in this step. For example, you specify a large genetype dataset but only used a small proportion of it in GWAS.

`--spark-conf` (required): spark configuration file.

#### Notes
**HEIG** is built on [hail](https://hail.is) for internal genetic analysis. `hail.MatrixTable` is the standard data format for genotype data. All processing based on it can benefit from distributed computing. We strongly encourage users to first convert other genotype data format such as `bfile` and `VCF` to `hail.MatrixTable` (Refer to Creating MatrixTable module), although `bfile` and `VCF` are supported by `--gwas`. This module supports a series of data management, including `--keep`, `--remove`, `--extract`, `--exclude`, `--chr-interval`, `--maf-min`, `--maf-max`, `--hwe`, `--variant-type`, `--call-rate` (Refer to Creating MatrixTable module).

**HEIG** currently cannot handle `VCF` files with no SNP rsID included.

An alternative option to select LDRs is `--ldr-col`, which specifies the LDR indices, e.g., `10,11,12` or equivalently `10:12`. Note this indices must be consecutive-`10,12,13` is not allowed.

We have tested the spark configuration file on computing clusters (not computing cloud such as AWS). For larger sample size, consider increasing `spark.executor.memory` and `spark.driver.memory` to `8g`, `12g` or `16g`. 

This module can also be used to conduct GWAS for any continuous non-imaging phenotypes.

#### Output
- `output/gwas/ldr_1kg.parquet`: LDR GWAS results in parquet format, which can be read into Python using pandas: `pd.read_parquet('output/gwas/ldr_1kg.parquet')`.


## 5. Processing summary statistics
* To process LDR GWAS summary statistics generated by **HEIG**:
```
heig.py \
--sumstats \
--out output/sumstats/ldr_1kg_top5 \
--ldr-gwas-heig output/gwas/ldr_1kg.parquet \
```
`--sumstats` (required): the flag for processing summary statistics.

`--out` (required): the prefix of output.

`--ldr-gwas-heig` (required): summary statistics generated by **HEIG**.


* To process LDR GWAS summary statistics generated by other software, such as PLINK2:
```
heig.py \
--sumstats \
--out output/sumstats/FA_SFO_phase123_white_unrel0.05 \
--threads 4 \
--ldr-gwas input/ldr_sumstats/FA_SFO_phase123_white_unrel0.05.{0:18}.glm.linear.gz \
--n-col N \
--chr-col CHR \
--pos-col BP \
--snp-col SNP \
--a1-col A1 \
--a2-col A2 \
--effect-col BETA,0 \
--se-col SE \
```
`--ldr-gwas` (required): LDR GWAS summary statistics using shortcut `{0:18}`, which means from `FA_SFO_phase123_white_unrel0.05.0.glm.linear.gz` to `FA_SFO_phase123_white_unrel0.05.18.glm.linear.gz`, 19 files in total.

`--n-col`: sample size column or `--n`: sample size as a integer, either one is required.

`--chr-col` (required): chromosome column.

`--pos-col` (required): position column.

`--snp-col` (required): SNP rsID column.

`--a1-col` (required): effective allele column.

`--a2-col`: (required): non-effective allele column.

`--effect-col` (required): genetic effect column along with the null value. Since imaging data is always continuous, the null values is always 0.

`--se-col` (required): standard error column.


* To process summary statistics of non-imaging phenotypes:
```
heig.py \
--sumstats \
--out output/sumstats/adhd \
--y2-gwas input/misc/ADHD2022_iPSYCH_deCODE_PGC_subset.meta \
--n-col N \
--snp-col SNP \
--a1-col A1 \
--a2-col A2 \
--effect-col OR,1 \
--se-col SE \
```
`--y2-gwas` (required): non-imaging phenotype summary statistics.

The following flags are slightly different from above. Non-imaging traits will only be used in cross-trait genetic correlation analysis. Therefore, only `SNP`, `A1`, `A2`, `N`, and `Z` are required. There are three different ways to get `Z`:

The following flags are slightly different from above. Non-imaging traits will only be used in cross-trait genetic correlation analysis. Therefore, only `SNP`, `A1`, `A2`, `N` and `Z` are required. There are three different ways to get `Z`:

1. Using `--z-col` to specify the Z score column.

2. Using `--effect-col` and `--se-col` to specify the genetic effect and standard error columns, respectively. If the effect is `BETA` or `logOR`, the null value is 0; if it is `OR`, then the null value is 1. `--se-col` must be the standard error of `BETA` or `logOR`.

3. Using `--effect-col` and `--p-vol` to specify the genetic effect and p-value columns, respectively. Note the p-value must not be the -log10 p-value.

#### Output
- `output/sumstats/ldr_1kg_top5.sumstats`: a HDF5 file of summary statistics (beta and z-score).
- `output/sumstats/ldr_1kg_top5.snpinfo`: a text file for SNP information.


## 6. Voxel-level GWAS reconstruction
* Scenario 1: screening the whole genome for all voxels.
```
heig.py \
--voxel-gwas \
--out output/voxelgwas/FA_SFO_phase123_white_unrel0.05_all_voxels_whole_genome \
--n-ldrs 19 \
--ldr-sumstats output/sumstats/FA_SFO_phase123_white_unrel0.05 \
--bases input/misc/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--ldr-cov input/misc/FA_SFO_phase123_white_unrel0.05_ldr_cov_top39.npy \
--threads 4 \
--sig-thresh 5.450185e-09 \
```
`--voxel-gwas` (required): the flag for voxel-level GWAS reconstruction.

`--out` (required): the prefix of output.

`--n-ldrs` (optional): the number of LDRs. Required if the number of LDRs differs in summary statistics and bases.

`--bases` (required): the functional bases.

`--ldr-cov` (required): the variance-covariance matrix of the covariate-effect-removed LDRs.

`--threads` (optional): the number of threads in parallel.

`--sig-thresh` (required): a significance threshold, which can either be a decimal `0.000000005450185` or in scientific notation `5.450185e-09`.

* Scenario 2: recovering whole-genome associations for a single voxel.
```
heig.py \
--voxel-gwas \
--out output/voxelgwas/FA_SFO_phase123_white_unrel0.05_whole_genome \
--n-ldrs 19 \
--ldr-sumstats output/sumstats/FA_SFO_phase123_white_unrel0.05 \
--bases input/misc/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--ldr-cov input/misc/FA_SFO_phase123_white_unrel0.05_ldr_cov_top39.npy \
--voxel {6:7} \
--threads 4 \
```
`--voxel` (required): a subset of voxels. Voxels should be one-based (the first one is `1`). It can be provided as a single number `--voxel 3`, or multiple numbers `--voxel 1,2,3`, `--voxel {1:10}`, which means voxel 1 to 10 (included), or a file `--voxel <filename>`.

* Scenario 3: Generating an atlas of associations for a few SNPs across all voxels.
```
heig.py \
--voxel-gwas \
--out output/voxelgwas/FA_SFO_phase123_white_unrel0.05 \
--n-ldrs 19 \
--ldr-sumstats output/sumstats/FA_SFO_phase123_white_unrel0.05 \
--bases input/misc/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--ldr-cov input/misc/FA_SFO_phase123_white_unrel0.05_ldr_cov_top39.npy \
--chr-interval 13:81000000,13:82000000 \
--threads 4 \
```
`--chr-interval` (optional): a segment of chromosome, which means from position `81000000` to `82000000` on the `chromosome 13`. All SNPs included in the segment were analyzed.

Alternatively, `--extract` and `--exclude` can be used to extract and exclude SNPs. Refer to Data Management.

#### Output
a text file of voxel-level GWAS results.

## 7. Heritability and (cross‐trait) genetic correlation analysis
1. Heritability and genetic correlation analysis within images.
```
heig.py \
--heri-gc \
--out output/herigc/FA_SFO_phase123_white_unrel0.05 \
--n-ldrs 19 \
--ldr-sumstats output/sumstats/FA_SFO_phase123_white_unrel0.05 \
--bases input/misc/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--ldr-cov input/misc/FA_SFO_phase123_white_unrel0.05_ldr_cov_top39.npy \
--threads 4 \
--ld-inv input/ld_regu8580/rand_10ksub_chr{1:22}_unrel_ld_inv_prop80 \
--ld input/ld_regu8580/rand_10ksub_chr{1:22}_unrel_ld_prop85 \
```
`--heri-gc` (required): the flag for heritability and genetic correlation analysis.

`--out` (required): the prefix of output.

`--n-ldrs` (optional): the number of LDRs. Required if the number of LDRs differs in summary statistics and bases.

`--ldr-sumstats` (required): prefix of processed LDR GWAS summary statistics.

`--bases` (required): the functional bases.

`--ldr-cov` (required): the variance-covariance matrix of the covariate-effect-removed LDRs.

`--threads` (optional): the number of threads in parallel.

`--ld-inv` (required):  the prefix of LD matrix inverse. Usually the LD matrix is separately estimated for each chromosome. Use the syntax `rand_10ksub_chr{1:22}_unrel_ld_inv_prop80` to represent `rand_10ksub_chr1_unrel_ld_inv_prop80` to `rand_10ksub_chr22_unrel_ld_inv_prop80`, 22 files in total.

`--ld` (required): the prefix of LD matrix.

#### Notes
Sometimes we are interested in only the heritability, then `--heri-only` can be used to skip genetic correlations.

We provided LD matrices at [Zenodo](https://zenodo.org/records/13787684) for genotyped data and imputed data, respectively. Choose one that matches the summary statistics.

#### Output
- `output/herigc/FA_SFO_phase123_white_unrel0.05_heri.txt`: a text file for heritability results.
- `output/herigc/FA_SFO_phase123_white_unrel0.05_gc.h5`: a HDF5 file for genetic correlation matrix.


2. Cross-trait genetic correlation between images and non-imaging traits.
```
heig.py \
--heri-gc \
--out output/herigc/FA_SFO_phase123_white_unrel0.05_adhd_overlap \
--n-ldrs 19 \
--ldr-sumstats output/sumstats/FA_SFO_phase123_white_unrel0.05 \
--bases input/misc/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--ldr-cov input/misc/FA_SFO_phase123_white_unrel0.05_ldr_cov_top39.npy \
--threads 4 \
--ld-inv input/ld_regu7570/rand_10ksub_chr{1:22}_unrel_ld_inv_prop70 \
--ld input/ld_regu7570/rand_10ksub_chr{1:22}_unrel_ld_prop75 \
--y2-sumstats output/sumstats/adhd \
--overlap \
```
`--y2-sumstats` (required): the prefix of processed GWAS summary statistics for non-imaging phenotypes.

`--overlap` (optional): a flag indicating potential sample overlap between two studies.

#### Output
- `output/herigc/FA_SFO_phase123_white_unrel0.05_adhd_overlap_gc.txt`: a text file for cross-trait genetic correlations.