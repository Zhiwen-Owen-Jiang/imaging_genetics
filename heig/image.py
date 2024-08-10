import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import heig.input.dataset as ds


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
            img_id = img_file.replace(suffix, '')
            if (img_file.endswith(suffix) and ((keep_idvs is not None and img_id in keep_idvs) or
                                               (keep_idvs is None))):
                if img_id in img_files:
                    n_dup += 1
                else:
                    img_files[img_id] = os.path.join(img_dir, img_file)
    img_files = dict(sorted(img_files.items()))
    ids = pd.MultiIndex.from_arrays([img_files.keys(), img_files.keys()], names=['FID', 'IID'])
    img_files_list = list(img_files.values())
    if n_dup > 0:
        log.info(f'WARNING: {n_dup} duplicated subjects. Keep the first appeared one.')

    return ids, img_files_list


def load_nifti(img_files):
    """
    Loading NifTi images.

    Parameters:
    ------------
    img_files: a list of image files

    Returns:
    ---------
    images (n, N): a np.array of imaging data
    coord (N, dim): a np.array of coordinates

    """
    try:
        img = nib.load(img_files[0])
    except:
        raise ValueError(('cannot read the image, did you provide FreeSurfer images '
                          'but forget to provide a Freesurfer surface mesh file?'))

    for i, img_file in enumerate(tqdm(img_files, desc=f'Loading {len(img_files)} images')):
        img = nib.load(img_file)
        data = img.get_fdata()
        if i == 0:
            idxs = data != 0
            coord = np.stack(np.nonzero(data)).T
            n_voxels = np.sum(idxs)
            images = np.zeros((len(img_files), n_voxels), dtype=np.float32)
        images[i] = data[idxs]

    return images, coord


def load_cifti(img_files, coord):
    """
    Loading CIFTI images.

    Parameters:
    ------------
    img_files: a list of image files

    Returns:
    ---------
    images (n, N): a np.array of imaging data

    """
    try:
        img = nib.load(img_files[0])
    except:
        raise ValueError(('cannot read the image, did you provide FreeSurfer images '
                          'but forget to provide a Freesurfer surface mesh file?'))

    for i, img_file in enumerate(tqdm(img_files, desc=f'Loading {len(img_files)} images')):
        img = nib.load(img_file)
        data = img.get_fdata()[0]
        if i == 0:
            n_voxels = len(data)
            if n_voxels != coord.shape[0]:
                raise ValueError('the CIFTI and GIFTI data has different number of vertices')
            images = np.zeros((len(img_files), n_voxels), dtype=np.float32)
        images[i] = data

    return images


def load_freesurfer(img_files, coord):
    """
    Loading freeSurfer outputs.

    Parameters:
    ------------
    img_files: a list of image files
    coord (N, dim): a np.array of coordinates 

    Returns:
    ---------
    images (n, N): a np.array of imaging data

    """
    for i, img_file in enumerate(tqdm(img_files, desc=f'Loading {len(img_files)} images')):
        data = nib.freesurfer.read_morph_data(img_file)
        if i == 0:
            images = np.zeros((len(img_files), len(data)), dtype=np.float32)
            if coord.shape[0] != images.shape[1]:
                raise ValueError(('the FreeSurfer surface mesh data and morphometry data '
                                  'have inconsistent coordinates'))
        if len(data) != images.shape[1]:
            raise ValueError('images have inconsistent resolution')
        images[i] = data

    return images


def save_images(out, images, coord, id):
    """
    Save imaging data to a HDF5 file.

    Parameters:
    ------------
    out: prefix of output
    images (n, N): a np.array of imaging data
    coord (N, dim): a np.array of coordinate 
    id: a pd.MultiIndex instance of IDs (FID, IID)

    """
    with h5py.File(f'{out}.h5', 'w') as file:
        dset = file.create_dataset('images', data=images)
        file.create_dataset('id', data=np.array(id.tolist(), dtype='S10'))
        file.create_dataset('coord', data=coord)
        dset.attrs['id'] = 'id'
        dset.attrs['coord'] = 'coord'


def check_input(args):
    ## --image-txt + --coord or --image-dir + --image-suffix is required
    if args.image_txt is not None:
        if args.coord is None:
            raise ValueError('--coord is required')
        if not os.path.exists(args.image_txt):
            raise FileNotFoundError(f"{args.image_txt} does not exist")
        if not os.path.exists(args.coord):
            raise FileNotFoundError(f"{args.coord} does not exist")
    elif args.image_dir is not None and args.image_suffix is None:
        raise ValueError('--image-suffix is required')
    elif args.image_dir is None and args.image_suffix is not None:
        raise ValueError('--image-dir is required')
    else:
        raise ValueError('--image-txt + --coord or --image-dir + --image-suffix is required')
    
    ## process arguments
    if args.image_dir is not None and args.image_suffix is not None:
        args.image_dir = args.image_dir.split(',')
        args.image_suffix = args.image_suffix.split(',')
        if len(args.image_dir) != len(args.image_suffix):
            raise ValueError('--image-dir and --image-suffix do not match')
        for image_dir in args.image_dir:
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"{image_dir} does not exist")
    if args.surface_mesh is not None and not os.path.exists(args.surface_mesh):
        raise FileNotFoundError(f"{args.surface_mesh} does not exist")
    if args.gifti is not None and not os.path.exists(args.gifti):
        raise FileNotFoundError(f"{args.gifti} does not exist")

    return args


def run(args, log):
    # check input
    args = check_input(args)

    # subjects to keep
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects are in --keep.')
    else:
        keep_idvs = None

    # read images
    if args.image_txt is not None:
        images = ds.Dataset(args.image_txt)
        log.info(f'{images.data.shape[0]} subjects and {images.data.shape[1]} voxels read from {args.image_txt}')
        if keep_idvs is not None:
            images.keep(keep_idvs)
            log.info(f'Keep {images.data.shape[0]} subjects.')
        ids = images.data.index

        coord = pd.read_csv(args.coord, sep='\s+', header=None)
        log.info(f'Read coordinate from {args.coord}')
        if coord.shape[0] != images.data.shape[1]:
            raise ValueError('images and coordinates have different resolution')
        images = np.array(images, dtype=np.float32)
    else:
        ids, img_files = get_image_list(args.image_dir, args.image_suffix, log, keep_idvs)
        if args.surface_mesh is not None:
            coord = nib.freesurfer.read_geometry(args.surface_mesh)[0]
            images = load_freesurfer(img_files, coord)
        elif args.gifti is not None:
            coord = nib.load(args.gifti).darrays[0].data
            images = load_cifti(img_files)
        else:
            images, coord = load_nifti(img_files)
        log.info((f"{images.shape[0]} subjects and {images.shape[1]} voxels '
                  'in the imaging data."))

    # save images
    save_images(args.out + '_images', images, coord, ids)
    log.info(f'Save the images to {args.out}_images.h5')
