# Highly-Efficient Imaging Genetics (HEIG)
HEIG is a statistical framework for efficiently conducting joint analysis for large-scale imaging and genetic data. Compared to traditional methods, HEIG reduces computational time and storage burden by over 100 times, significantly boosts statistical power in association analysis, and most importantly, defines the standard to share the voxel-level GWAS summary statistics to the community. 

The analysis can be performed by HEIG (will have more in the near future):
- Voxelwise genome-wide association analysis (VGWAS)
- Voxelwise heritability analysis
- Genetic correlation analysis for pairs of voxels
- Cross-trait genetic correlation between voxels and non-imaging phenotypes

## Getting Started
First clone the repository using the commands
```
git clone https://github.com/Zhiwen-Owen-Jiang/heig.git
cd heig
```
You can use [Anaconda](https://www.anaconda.com) to build the environment and install all dependencies for HEIG using the commands
```
conda env create --file environment.yml
conda activate heig
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
