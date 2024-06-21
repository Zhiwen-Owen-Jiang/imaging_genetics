# Highly Efficient Imaging Genetics (HEIG)
HEIG is a statistical framework for efficiently conducting joint analysis for large-scale imaging and genetic data. Compared to traditional methods, HEIG reduces computational time and storage burden by over 100 times, significantly boosts statistical power in association analysis, and most importantly, defines the standard to share the voxel-level GWAS summary statistics to the community. 

The analysis can be performed by HEIG (will have more in the near future):
- Voxelwise genome-wide association analysis (VGWAS)
- Voxelwise heritability analysis
- Genetic correlation analysis for pairs of voxels
- Cross-trait genetic correlation between voxels and non-imaging phenotypes

## Version
- [v1.0.0](https://github.com/Zhiwen-Owen-Jiang/heig/releases/tag/v1.0.0): The initial version of HEIG.

## System Requirements
### OS Requirements
HEIG is supported for macOS and Linux. It has been tested on the following systems:
- Red Hat Enterprise Linux 8.9
- MacOS Sonoma 14.5

### Python Dependencies
We implement HEIG on Python 3.11. Specific package dependencies are provided in [requirements](https://github.com/Zhiwen-Owen-Jiang/heig/blob/pub/requirements.txt).

## Getting Started
First download the [released version](https://github.com/Zhiwen-Owen-Jiang/heig/releases/tag/v1.0.0), unzip it, and navigate to the extracted folder:
```
wget -O heig-1.0.0.zip  https://github.com/Zhiwen-Owen-Jiang/heig/archive/refs/tags/v1.0.0.zip
unzip heig-1.0.0.zip
cd heig-1.0.0
```
Install [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) based on your OS, and build the environment and install all dependencies for HEIG. This step may take ~5 minutes.
```
conda env create --file environment.yml
conda activate heig
```
Or you can do it manually. 

## How to use HEIG
We provided detailed [tutorial](https://github.com/Zhiwen-Owen-Jiang/heig/wiki) for using HEIG. The example data used in the tutorial can be downloaded [here](https://zenodo.org/records/11075259). Common issues are described in the [FAQ](https://github.com/Zhiwen-Owen-Jiang/heig/wiki/FAQ).

If that does not work, email Owen Jiang <owenjf@live.unc.edu> or <zhiwenowenjiang@gmail.com>.

## Citation
TBD.

## Licence
This project is licensed under GNU GPL v3.

## Authors
Zhiwen (Owen) Jiang and Hongtu Zhu (University of North Carolina at Chapel Hill)
