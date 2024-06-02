## setup
# Download HEIG v1.0.0 at https://github.com/Zhiwen-Owen-Jiang/heig/releases/tag/v1.0.0, 
# and follow instructions at https://github.com/Zhiwen-Owen-Jiang/heig to build up the environment.
# The shared data for replicating the results are at https://doi.org/10.5281/zenodo.11404334

## code for voxelwise GWAS
python heig.py \
--voxel-gwas \
--n-ldrs 19 \
--ldr-sumstats shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_top19_6msnp \
--bases shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--inner-ldr shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy \
--sig-thresh 1.91e-10 \
--out SFO_voxel_gwas

## code for heritability and genetic correlation analysis
python heig.py \
--heri-gc \
--n-ldrs 19 \
--ldr-sumstats shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_top19_6msnp \
--bases shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--inner-ldr shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy \
--ld-inv shared_data/ldmatrix_hapmap3/ld_regu9895/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_inv_prop95 \
--ld shared_data/ldmatrix_hapmap3/ld_regu9895/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_prop98 \
--out SFO_heri_gc

## code for preprocessing summary statistics
## download insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt.gz from https://vu.data.surfsara.nl/index.php/s/06RsHECyWqlBRwq
## and unzip it by
gunzip insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt.gz

python heig.py \
--sumstats \
--y2-gwas insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt \
--n-col NMISS \
--snp-col RSID_UKB \
--a1-col A1 \
--a2-col A2 \
--z STAT \
--out insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100

## code for cross-trait genetic correlation analysis
python heig.py \
--heri-gc \
--n-ldrs 19 \
--ldr-sumstats shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_top19_6msnp \
--bases shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--inner-ldr shared_data/ukb_wm_microstructure_phase123_white_unrel0.05/SFO/FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy \
--ld-inv shared_data/ldmatrix_hapmap3/ld_regu9085/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_inv_prop85 \
--ld shared_data/ldmatrix_hapmap3/ld_regu9085/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_prop90 \
--y2-sumstats insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100 \
--overlap \
--out SFO_insomnia
