## Gettting started
In this tutorial, we will go through seven analysis modules in **HEIG** (v.1.1.0), including reading images, functional PCA, constructing LDRs, preprocessing summary statistics, voxelwise GWAS reconstruction, heritability and (cross-trait) genetic correlation analysis, and LD matrix construction. For beginners, we recommend reading these modules in the above sequence. 

To replicate the analyses showed in the tutorial, you can download the example data at [Zenodo](https://zenodo.org/records/13770930). The total file size is 1.1 GB.

After downloading the example data, unzip it and navigate to it. Make sure your current directory is `example`. Let's get started.


## Reading images
```
heig.py \
--read-image \
--out output/images/example \
--threads 4 \
--image-dir input/images/ \
--image-suffix _example_image.nii.gz \
--coord-dir input/images/s1000_example_image.nii.gz \
```
`--read-image` is the main flag for reading images and `--out` specifies the prefix of output. 

`--threads` is to specify the number of threads in parallel.

`--image-dir` is to specify the directory(s) and flag `--image-suffix` to specify the suffix(s). **HEIG** supports images in `NIFTI`, `CIFTI2`, and FreeSurfer morphometry data format. Prior to loading into **HEIG**, all images should be appropriately registered/projected to the same template. Images can be placed under one or more directories. Separate multiple directories by comma, such as `data/image_folder1,data/image_folder2,data/image_folder3`. The naming convention of image file is `<ID><suffix>`. For example, images in this tutorial were named as `sxxxx_example_image.nii.gz`, where `sxxxx` (e.g., s1001) is the ID and `_example_image.nii.gz` is the suffix. If images are from different directories, the same number of suffices must be provided and separated by comma, such as `suffix1,suffix2,suffix3`. 

`--coord-dir` is to specify the coordinate file. For `NIFTI` images, it should also be a `NIFTI` image; for `CIFTI2` images, it should be a `GIFTI` image; for FreeSurfer morphometry data, it should be a FreeSurfer surface mesh file. Only one coordinate file can be provided even if you have images from multiple directories.

**HEIG** also supports images in tabular data. Using `--image-txt` and `--coord-txt` to read images and coordinates, respectively. Refer to [Basic options and input formats](https://github.com/Zhiwen-Owen-Jiang/heig/wiki/Basic-options-and-input-formats) for detailed format.

