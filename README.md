# Highly-Efficient Imaging Genetics (HEIG)
HEIG is a statistical framework for efficiently conducting joint analysis for large-scale imaging and genetic data. Currently, HEIG is capable of performing voxelwise genome-wide association analysis (GWAS) on images for unrelated subjects, as well as heritability analysis for each voxel, genetic correlation analysis for a pair of voxels, and cross-trait genetic correlation between a voxel and non-imaging phenotype using summary statistics. 

## Getting Started
First clone the repository using the commands
```
git clone https://github.com/Zhiwen-Owen-Jiang/heig.git
cd heig
```
You can use [Anaconda](https://www.anaconda.com) to build the environment and install all dependencies for HEIG using the commands
```
conda env create --file environment.yml
source activate heig
```
Or you can do it manually. Using the following command to test if the enviroment was successfully built
```
python heig.py -h
```
If it failed with an error, then go back to check if all requirements are installed and if the version is compatible.

## Updating HEIG
HEIG is under active update and more analysis modules will be added in the near future. To update HEIG using the command
```
git pull
```

## How to use HEIG
We provided detailed [tutorial](https://github.com/Zhiwen-Owen-Jiang/heig/wiki) for using HEIG. The example data used in the tutorial can be downloaded [here](https://zenodo.org/records/11075259). Common issues are described in the [FAQ](https://github.com/Zhiwen-Owen-Jiang/heig/wiki/FAQ).

If that does not work, email Owen Jiang <owenjf@live.unc.edu> or <zhiwenowenjiang@gmail.com>.

## Citation
TBD.

## Licence
This project is licensed under GNU GPL v3.

## Authors
Zhiwen (Owen) Jiang and Hongtu Zhu (University of North Carolina at Chapel Hill)
