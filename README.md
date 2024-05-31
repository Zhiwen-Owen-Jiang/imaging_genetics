# Highly-Efficient Imaging Genetics (HEIG)
HEIG is a statistical framework for efficiently conducting joint analysis for large-scale imaging and genetic data. Compared to traditional methods, HEIG reduces computational time and storage burden by over 100 times, significantly boosts statistical power in association analysis, and most importantly, defines the standard to share the voxel-level GWAS summary statistics to the community. 

The analysis can be performed by HEIG (will have more in the near future):
- Voxelwise genome-wide association analysis (VGWAS)
- Voxelwise heritability analysis
- Genetic correlation analysis for pairs of voxels
- Cross-trait genetic correlation between voxels and non-imaging phenotypes

## Version
[v1.0.0](https://github.com/Zhiwen-Owen-Jiang/heig/releases/tag/v1.0.0): the initial version of HEIG.

## Getting Started
First download the [released version](https://github.com/Zhiwen-Owen-Jiang/heig/releases/tag/v1.0.0), unzip it, and navigate to the extracted folder.
Alternatively, you can clone the repository and navigate to it using the commands
```
git clone https://github.com/Zhiwen-Owen-Jiang/heig.git
cd heig
```
Install [Anaconda](https://www.anaconda.com) to build the environment and install all dependencies for HEIG using the commands
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
