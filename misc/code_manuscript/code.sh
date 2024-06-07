# Setup
## Downloading HEIG v1.0.0 at https://github.com/Zhiwen-Owen-Jiang/heig/releases/tag/v1.0.0, 
## and following instructions at https://github.com/Zhiwen-Owen-Jiang/heig to build up the environment.
## The shared data for reproducing the results is at https://doi.org/10.5281/zenodo.11404334.
## We take SFO for example. Downloading the data and unzipping it by
wget -O ukb_wm_microstructure_phase123_white_unrel0.05_SFO.tar.gz https://zenodo.org/records/11404334/files/ukb_wm_microstructure_phase123_white_unrel0.05_SFO.tar.gz?download=1
wget -O ldmatrix_hapmap3.tar.gz https://zenodo.org/records/11404334/files/ldmatrix_hapmap3.tar.gz?download=1
tar -xvzf ukb_wm_microstructure_phase123_white_unrel0.05_SFO.tar.gz
tar -xvzf ldmatrix_hapmap3.tar.gz

## You will get a folder SFO which contains all shared data. Specifically:
## - coord_FA_SFO.txt: coordinates of original image for visualization
## - FA_SFO_phase123_white_unrel0.05_bases_top39.npy: the top 39 functional bases
## - FA_SFO_phase123_white_unrel0.05_eigenvalues_top193.npy: the all 193 eigenvalues 
## - FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy: the inner project of covariate-effect-removed LDRs
## - FA_SFO_phase123_white_unrel0.05_top19_6msnp.snpinfo: a txt file of SNP information for summary statistics
## - FA_SFO_phase123_white_unrel0.05_top19_6msnp.sumstats: preprocessed LDR summary statistics for 19 LDRs
## - SFO.nii.gz: the ROI mask for visualization

## Addtionally, you will get a folder ldmatrix_hapmap3, 
## which contains LD matrix and its inverse for 22 chromosomes, under two different combinations of regularization {98%,95%} and {90%,85%}


# Code for voxelwise GWAS
## For other ROIs, --n-ldrs can be specified based on the LDR summary statistics file
## --n-ldrs is 19 for this case
python heig.py \
--voxel-gwas \
--n-ldrs 19 \
--ldr-sumstats SFO/FA_SFO_phase123_white_unrel0.05_top19_6msnp \
--bases SFO/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--inner-ldr SFO/FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy \
--sig-thresh 1.91e-10 \
--out SFO_voxel_gwas


# Code for heritability and genetic correlation analysis
python heig.py \
--heri-gc \
--n-ldrs 19 \
--ldr-sumstats SFO/FA_SFO_phase123_white_unrel0.05_top19_6msnp \
--bases SFO/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--inner-ldr SFO/FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy \
--ld-inv ldmatrix_hapmap3/ld_regu9895/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_inv_prop95 \
--ld ldmatrix_hapmap3/ld_regu9895/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_prop98 \
--out SFO_heri_gc


# Code for preprocessing summary statistics
## Downloading insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt.gz from https://vu.data.surfsara.nl/index.php/s/06RsHECyWqlBRwq
## and unzipping it by
gunzip insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt.gz

## Checking the first two rows to get column names
head -n 2 insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt

## Preprocessing summary statistics
python heig.py \
--sumstats \
--y2-gwas insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt \
--n-col NMISS \
--snp-col RSID_UKB \
--a1-col A1 \
--a2-col A2 \
--z STAT \
--out insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100


# Code for cross-trait genetic correlation analysis
python heig.py \
--heri-gc \
--n-ldrs 19 \
--ldr-sumstats SFO/FA_SFO_phase123_white_unrel0.05_top19_6msnp \
--bases SFO/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--inner-ldr SFO/FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy \
--ld-inv ldmatrix_hapmap3/ld_regu9085/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_inv_prop85 \
--ld ldmatrix_hapmap3/ld_regu9085/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_prop90 \
--y2-sumstats insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100 \
--overlap \
--out SFO_insomnia
