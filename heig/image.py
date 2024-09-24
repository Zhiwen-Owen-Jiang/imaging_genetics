import os
import logging
import h5py
import concurrent.futures
from abc import ABC, abstractmethod
from filelock import FileLock
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
import heig.input.dataset as ds


class ImageReader(ABC):
    """
    An abstract class for reading images

    """

    def __init__(self, img_files, ids, out_dir):
        self.img_files = img_files
        self.n_images = len(self.img_files)
        self.ids = ids
        self.out_dir = out_dir
        self.logger = logging.getLogger(__name__)

    def create_dataset(self, coord_img_file):
        """
        Creating a HDF5 file saving images, coordinates, and ids

        """
        self.coord = self._get_coord(coord_img_file)
        self.n_voxels = self.coord.shape[0]

        with h5py.File(self.out_dir, "w") as h5f:
            images = h5f.create_dataset(
                "images", shape=(self.n_images, self.n_voxels), dtype="float32"
            )
            h5f.create_dataset("id", data=np.array(self.ids.tolist(), dtype="S10"))
            h5f.create_dataset("coord", data=self.coord)
            images.attrs["id"] = "id"
            images.attrs["coord"] = "coord"
        self.logger.info(
            (
                f"{self.n_images} subjects and {self.n_voxels} voxels (vertices) "
                "in the imaging data."
            )
        )

    def read_save_image(self, threads):
        """
        Reading and saving images in parallel

        """
        self.logger.info("Reading images ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(self._read_save_image, idx, img_file)
                for idx, img_file in enumerate(self.img_files)
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{len(futures)} images",
            ):
                pass

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    self.logger.info(f"Generated an exception: {exc}.")
                    break

        self.logger.info("Done.")
        if os.path.exists(f"{self.out_dir}.lock"):
            os.remove(f"{self.out_dir}.lock")

    def _read_save_image(self, idx, img_file):
        """
        Reading and writing a single image

        """
        image = self._read_image(img_file)
        if len(image) != self.n_voxels:
            raise ValueError(
                f"{img_file} is of resolution {len(image)} but the coordinate is of resolution {self.n_voxels}"
            )
        lock_file = f"{self.out_dir}.lock"
        with FileLock(lock_file):
            with h5py.File(self.out_dir, "r+") as h5f:
                h5f["images"][idx] = image

    @abstractmethod
    def _get_coord(self, coord_img_file):
        pass

    @abstractmethod
    def _read_image(self, img_file):
        pass


class NIFTIReader(ImageReader):
    """
    Reading NIFTI images and coordinates.

    """

    def _get_coord(self, coord_img_file):
        img = nib.load(coord_img_file)
        data = img.get_fdata()
        coord = np.stack(np.nonzero(data)).T
        return coord

    def _read_image(self, img_file):
        try:
            img = nib.load(img_file)
            data = img.get_fdata()
            data = data[tuple(self.coord.T)]
            return data
        except:
            raise ValueError(
                "cannot read the image, did you provide a wrong NIFTI image?"
            )


class CIFTIReader(ImageReader):
    """
    Reading CIFTI images and coordinates.

    """

    def _get_coord(self, coord_img_file):
        """
        Reading coordinates from a GIFTI image.

        """
        coord = nib.load(coord_img_file).darrays[0].data
        return coord

    def _read_image(self, img_file):
        try:
            img = nib.load(img_file)
            data = img.get_fdata()[0]
            return data
        except:
            raise ValueError(
                "cannot read the image, did you provide a wrong CIFTI image?"
            )


class FreeSurferReader(ImageReader):
    """
    Reading FreeSurfer outputs and coordinates.

    """

    def _get_coord(self, coord_img_file):
        """
        Reading coordinates from a Freesurfer surface mesh file

        """
        coord = nib.freesurfer.read_geometry(coord_img_file)[0]
        return coord

    def _read_image(self, img_file):
        try:
            data = nib.freesurfer.read_morph_data(img_file)
            return data
        except:
            raise ValueError(
                (
                    "cannot read the image, did you provide a wrong "
                    "FreeSurfer morphometry data file?"
                )
            )


def get_image_list(img_dirs, suffixes, log, keep_idvs=None):
    """
    Getting file path of images from multiple directories.

    Parameters:
    ------------
    img_dirs: a list of directories
    suffixes: a list of suffixes of images
    log: a logger
    keep_idvs: a pd.MultiIndex instance of IDs (FID, IID)

    Returns:
    ---------
    ids: a pd.MultiIndex instance of IDs
    img_files_list: a list of image files to read

    """
    img_files = {}
    n_dup = 0

    for img_dir, suffix in zip(img_dirs, suffixes):
        for img_file in os.listdir(img_dir):
            img_id = img_file.replace(suffix, "")
            if img_file.endswith(suffix) and (
                (keep_idvs is not None and img_id in keep_idvs) or (keep_idvs is None)
            ):
                if img_id in img_files:
                    n_dup += 1
                else:
                    img_files[img_id] = os.path.join(img_dir, img_file)
    img_files = dict(sorted(img_files.items()))
    ids = pd.MultiIndex.from_arrays(
        [img_files.keys(), img_files.keys()], names=["FID", "IID"]
    )
    img_files_list = list(img_files.values())
    if n_dup > 0:
        log.info(f"WARNING: {n_dup} duplicated subjects. Keep the first appeared one.")

    return ids, img_files_list


def save_images(out_dir, images, coord, id):
    """
    Save imaging data to a HDF5 file.

    Parameters:
    ------------
    out_dir: prefix of output
    images (n, N): a np.array of imaging data
    coord (N, dim): a np.array of coordinate
    id: a pd.MultiIndex instance of IDs (FID, IID)

    """
    with h5py.File(out_dir, "w") as file:
        dset = file.create_dataset("images", data=images, dtype="float32")
        file.create_dataset("id", data=np.array(id.tolist(), dtype="S10"))
        file.create_dataset("coord", data=coord)
        dset.attrs["id"] = "id"
        dset.attrs["coord"] = "coord"


def check_input(args):
    ## --image-txt + --coord or --image-dir + --image-suffix is required
    if (
        args.image_dir is None
        and args.image_suffix is None
        and args.image_txt is None
        and args.coord_txt is None
    ):
        raise ValueError(
            "--image-txt + --coord-txt or --image-dir + --image-suffix is required"
        )

    if args.image_txt is not None and args.coord_txt is None:
        raise ValueError("--coord-txt is required")
    elif args.image_txt is None and args.coord_txt is not None:
        raise ValueError("--image-txt is required")
    elif args.image_dir is not None and args.image_suffix is None:
        raise ValueError("--image-suffix is required")
    elif args.image_dir is None and args.image_suffix is not None:
        raise ValueError("--image-dir is required")

    ds.check_existence(args.image_txt)
    ds.check_existence(args.coord_txt)

    ## process arguments
    if args.image_dir is not None and args.image_suffix is not None:
        if args.coord_dir is None:
            raise ValueError("--coord-dir is required")
        args.image_dir = args.image_dir.split(",")
        args.image_suffix = args.image_suffix.split(",")
        if len(args.image_dir) != len(args.image_suffix):
            raise ValueError("--image-dir and --image-suffix do not match")
        for image_dir in args.image_dir:
            ds.check_existence(image_dir)
        ds.check_existence(args.coord_dir)


def run(args, log):
    # check input
    check_input(args)

    # read images
    out_dir = f"{args.out}_images.h5"
    if args.image_txt is not None:
        images = ds.Dataset(args.image_txt)
        log.info(
            (
                f"{images.data.shape[0]} subjects and {images.data.shape[1]} "
                f"voxels (vertices) read from {args.image_txt}"
            )
        )
        if args.keep is not None:
            common_ids = ds.get_common_idxs(images.data.index, args.keep)
            images.keep(common_ids)
            log.info(f"Keep {images.data.shape[0]} subjects.")
        ids = images.data.index
        images = np.array(images.data, dtype=np.float32)

        coord = pd.read_csv(args.coord_txt, sep="\s+", header=None)
        log.info(f"Read coordinates from {args.coord_txt}")
        if coord.isnull().sum().sum() > 0:
            raise ValueError("no missing values allowed in coordinates")
        if coord.shape[0] != images.shape[1]:
            raise ValueError("images and coordinates have different resolution")
        save_images(out_dir, images, coord, ids)
    else:
        ids, img_files = get_image_list(
            args.image_dir, args.image_suffix, log, args.keep
        )
        if len(img_files) == 0:
            raise ValueError(
                f"no image in {args.image_dir} with suffix {args.image_suffix}"
            )
        if args.coord_dir.endswith("nii.gz") or args.coord_dir.endswith("nii"):
            log.info("Reading NIFTI images.")
            img_reader = NIFTIReader(img_files, ids, out_dir)
        elif args.coord_dir.endswith("gii.gz") or args.coord_dir.endswith("gii"):
            log.info("Reading CIFTI images.")
            img_reader = CIFTIReader(img_files, ids, out_dir)
        else:
            log.info("Reading FreeSurfer morphometry data.")
            img_reader = FreeSurferReader(img_files, ids, out_dir)
        img_reader.create_dataset(args.coord_dir)
        img_reader.read_save_image(args.threads)

    log.info(f"\nSave the images to {out_dir}")
