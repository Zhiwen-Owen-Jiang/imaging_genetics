### Prerequisite

Disk space of 16 GB is required for this analysis. A macOS or Linux machine with at least 8 GB RAM is required.

### Setup

Downloading HEIG v1.0.0 at https://github.com/Zhiwen-Owen-Jiang/heig/releases/tag/v1.0.0, and following instructions at https://github.com/Zhiwen-Owen-Jiang/heig to build up the environment. After that, make sure your current directory is `heig-1.0.0`. 

The shared data for reproducing the results is at https://doi.org/10.5281/zenodo.11404334. We take SFO for example. Downloading the data and unzipping it by

```
wget -O ukb_wm_microstructure_phase123_white_unrel0.05_SFO.tar.gz 'https://zenodo.org/records/11404334/files/ukb_wm_microstructure_phase123_white_unrel0.05_SFO.tar.gz?download=1'
wget -O ldmatrix_hapmap3.tar.gz 'https://zenodo.org/records/11404334/files/ldmatrix_hapmap3.tar.gz?download=1'
tar -xvzf ukb_wm_microstructure_phase123_white_unrel0.05_SFO.tar.gz
tar -xvzf ldmatrix_hapmap3.tar.gz
```

We will get a folder `SFO` which contains all shared data. Specifically:

- coord_FA_SFO.txt: coordinates of original image for visualization (193 voxels in total)
- FA_SFO_phase123_white_unrel0.05_bases_top39.npy: the top 39 functional bases
- FA_SFO_phase123_white_unrel0.05_eigenvalues_top193.npy: 193 eigenvalues 
- FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy: the inner project of top 39 covariate-effect-removed LDRs
- FA_SFO_phase123_white_unrel0.05_top19_6msnp.snpinfo: a txt file of SNP information for summary statistics (6.3 million)
- FA_SFO_phase123_white_unrel0.05_top19_6msnp.sumstats: preprocessed LDR summary statistics for top 19 LDRs
- SFO.nii.gz: the ROI mask for visualization

Addtionally, we will get a folder `ldmatrix_hapmap3`, which contains the LD matrix and its inverse for 22 chromosomes, under two different combinations of regularization {98%,95%} and {90%,85%}.

### Code for voxelwise GWAS

Since we only conducted GWAS for the top 19 LDRs (cumulatively contributed 80% of variance), we set `--n-ldrs` as 19. For other ROIs, `--n-ldrs` can be specified based on the LDR summary statistics file.

```
python heig.py \
--voxel-gwas \
--n-ldrs 19 \
--ldr-sumstats SFO/FA_SFO_phase123_white_unrel0.05_top19_6msnp \
--bases SFO/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--inner-ldr SFO/FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy \
--sig-thresh 1.91e-10 \
--out SFO_voxel_gwas
```

This analysis will produce `SFO_insomnia.log` and `SFO_voxel_gwas.txt` for the log and the results, repectively. We provided them at https://github.com/Zhiwen-Owen-Jiang/heig/tree/pub/misc/code_manuscript.

### Code for heritability and genetic correlation analysis

```
python heig.py \
--heri-gc \
--n-ldrs 19 \
--ldr-sumstats SFO/FA_SFO_phase123_white_unrel0.05_top19_6msnp \
--bases SFO/FA_SFO_phase123_white_unrel0.05_bases_top39.npy \
--inner-ldr SFO/FA_SFO_phase123_white_unrel0.05_proj_innerldr_top39.npy \
--ld-inv ldmatrix_hapmap3/ld_regu9895/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_inv_prop95 \
--ld ldmatrix_hapmap3/ld_regu9895/ukb_white_exclude_phase123_50k_sub_chr{1:22}_unrel_ld_prop98 \
--out SFO_heri_gc
```

This analysis will produce `SFO_heri_gc.log`, `SFO_heri_gc_heri.txt`, and `SFO_heri_gc_gc.npz` for the log, heritability, and genetic correlation results, repectively. We provided them at https://github.com/Zhiwen-Owen-Jiang/heig/tree/pub/misc/code_manuscript. We can compare the heritability results in `SFO_heri_gc.log` (mean, median, min, max, and mean se) to the results at line 20, Table S7. We can compare the genetic correlation results in `SFO_heri_gc.log` (mean, median, min, and mean se) to the results at line 20, Table S8. We may find there is slight difference in genetic correlation results, which is because `SFO_heri_gc.log` presents the summary results before removing out-of-bound results. Nevertheless, we can always use the following code to exactly reproduce the results

```python
import numpy as np

def get_2d_matrix(n_voxels, tril, gc=True):
    """
    Recovering the full matrix from the lower triangle part w/o the diagonal
    
    Parameters:
    ____________
    n_voxels: the number of voxels in the original image
    tril: the lower triangle part w/o the diagonal
    gc: True: it is a genetic correlation matrix; False: it is a standard error matrix

    Returns:
    _________
    matrix: a n_voxels by n_voxels symmetric matrix of genetic correlation between voxels

    """
    matrix = np.zeros((n_voxels, n_voxels))
    matrix[np.tril_indices(n_voxels, k = -1)] = tril
    matrix = matrix + matrix.T
    if gc:
        np.fill_diagonal(matrix, 1)
    else:
        np.fill_diagonal(matrix, 0)
    return matrix

data = np.load('SFO_heri_gc_gc.npz')
gc = get_2d_matrix(193, data['gc'], gc=True) # in total 193 voxels
se = get_2d_matrix(193, data['se'], gc=False) # in total 193 voxels
print(f'Mean GC: {np.round(np.nanmean(gc), 4)}')
print(f'Median GC: {np.round(np.nanmedian(gc), 4)}')
print(f'Min GC: {np.round(np.nanmin(gc), 4)}')
print(f'SD GC: {np.round(np.nanstd(gc), 4)}')
print(f'SE GC: {np.round(np.nanmean(se), 4)}')
```

### Code for preprocessing summary statistics

Manually downloading `insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt.gz` from https://vu.data.surfsara.nl/index.php/s/06RsHECyWqlBRwq and unzipping it by

```
gunzip insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt.gz
```

We check the first two rows to get column names

```
head -n 2 insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt
```
Then we preprocess the summary statistics

```
python heig.py \
--sumstats \
--y2-gwas insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.txt \
--n-col NMISS \
--snp-col RSID_UKB \
--a1-col A1 \
--a2-col A2 \
--z STAT \
--out insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100
```

This analysis will produce `insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.log`, `insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.snpinfo` and `insomnia_ukb2b_EUR_sumstats_20190311_with_chrX_mac_100.sumstats`. The first few rows for the `.snpinfo` file should be 

```
SNP A1  A2  N
rs570371753 C   T   386986
rs554639997 A   G   386987
rs12184267  T   C   381646
rs12184277  G   A   382038
rs12184279  A   C   382066
```
We provided the `.log` file at https://github.com/Zhiwen-Owen-Jiang/heig/tree/pub/misc/code_manuscript.

### Code for cross-trait genetic correlation analysis

```
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
```

This analysis will produce `SFO_insomnia.log` and `SFO_insomnia_gc.txt`. We provided them at https://github.com/Zhiwen-Owen-Jiang/heig/tree/pub/misc/code_manuscript. We can compare the cross-trait genetic correlation results in `SFO_insomnia.log` to the results at line 121, Table S11.
