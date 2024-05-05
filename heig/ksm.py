import os
import logging
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from numpy.linalg import inv
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, hstack
import heig.input.dataset as ds


class KernelSmooth:
    def __init__(self, data, coord):
        """
        Parameters:
        ------------
        data (n, N): raw imaging data
        coord (N, dim): coordinates

        """
        self.data = data
        self.coord = coord
        self.n, self.N = self.data.shape
        self.d = self.coord.shape[1]

    def _gau_kernel(self, x):
        """
        Calculating the Gaussian density

        Parameters:
        ------------
        x: a np.array of coordinates

        Returns:
        ---------
        gau_k: Gaussian density

        """
        gau_k = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)

        return gau_k

    def smoother(self):
        raise NotImplementedError

    def gcv(self, bw_list, log):
        """
        Generalized cross-validation for selecting the optimal bandwidth

        Parameters:
        ------------
        bw_list: a array of candidate bandwidths 
        log: a logger 

        Returns:
        ---------
        bw_opt: the optimal bandwidth

        """
        score = np.zeros(len(bw_list))

        for cii, bw in enumerate(bw_list):
            log.info(
                f"Doing generalized cross-validation (GCV) for bandwidth {np.round(bw, 3)} ...")
            isvalid, y_sm, sparse_sm_weight = self.smoother(bw)
            if isvalid:
                score[cii] = (np.mean(np.sum((self.data - y_sm) ** 2, axis=1)) /
                              (1 - np.sum(sparse_sm_weight.diagonal()) / self.N + 10**-10) ** 2)
                if score[cii] == 0:
                    score[cii] = np.nan
                log.info(
                    f"The GCV score for bandwidth {np.round(bw, 3)} is {round(score[cii], 3)}.")
            else:
                score[cii] = np.Inf

        which_min = np.nanargmin(score)
        if which_min == 0 or which_min == len(bw_list) - 1:
            log.info(("WARNING: the optimal bandwidth was obtained at the boundary, "
                      "which may not be the best one."))
        bw_opt = bw_list[which_min]
        min_mse = score[which_min]
        log.info(
            f"The optimal bandwidth is {np.round(bw_opt, 3)} with GCV score {round(min_mse, 3)}.")

        return bw_opt

    def bw_cand(self):
        """
        Generating a array of candidate bandwidths

        """
        bw_raw = self.N ** (-1 / (4 + self.d))
        weights = [0.2, 0.5, 1, 2, 5, 10]
        bw_list = np.zeros((len(weights), self.d))

        for i, weight in enumerate(weights):
            bw_list[i, :] = np.repeat(weight * bw_raw, self.d)

        return bw_list


class LocalLinear(KernelSmooth):
    def __init__(self, data, coord):
        super().__init__(data, coord)
        self.logger = logging.getLogger(__name__)

    def smoother(self, bw):
        """
        Local linear smoother. 

        Parameters:
        ------------
        bw (dim, 1): bandwidth for dim dimension

        Returns:
        ---------
        True/False: if the bandwidth is able to make the weight matrix sparse
        sm_data (n, N): smooothed imaging data
        sparse_sm_weight (N, N): sparse kernel smoothing weights

        """
        sparse_sm_weight = dok_matrix((self.N, self.N), dtype=np.float32)
        for lii in range(self.N):
            t_mat0 = self.coord - self.coord[lii]  # N * d
            t_mat = np.hstack((np.ones(self.N).reshape(-1, 1), t_mat0))
            dis = t_mat0 / bw
            close_points = (dis < 4) & (dis > -4)  # keep only nearby voxels
            k_mat = csr_matrix((self._gau_kernel(dis[close_points]), np.where(close_points)),
                               (self.N, self.d))
            k_mat = csc_matrix(np.prod(k_mat / bw, axis=1))  # can be faster
            k_mat_sparse = hstack([k_mat] * (self.d + 1))
            kx = k_mat_sparse.multiply(t_mat).T  # (d+1) * N
            sm_weight = inv(kx @ t_mat + np.eye(self.d + 1)
                            * 0.000001)[0, :] @ kx  # N * 1
            large_weight_idxs = np.where(np.abs(sm_weight) > 1 / self.N)
            sparse_sm_weight[lii,
                             large_weight_idxs] = sm_weight[large_weight_idxs]
        nonzero_weights = np.sum(sparse_sm_weight != 0, axis=0)
        if np.mean(nonzero_weights) > self.N // 10:
            self.logger.info((f"On average, the non-zero weight for each voxel "
                              f"are greater than {self.N // 10}. "
                              "Skip this bandwidth."))
            return False, None, None

        sm_data = self.data @ sparse_sm_weight.T
        return True, sm_data, sparse_sm_weight


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
    ids = pd.MultiIndex.from_arrays(
        [img_files.keys(), img_files.keys()], names=['FID', 'IID'])
    img_files_list = list(img_files.values())
    if n_dup > 0:
        log.info(
            f'WARNING: {n_dup} duplicated subjects. Keep the first appeared one.')

    return ids, img_files_list


def load_nifti(img_files):
    """
    Load NifTi images.
    
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


def load_freesurfer(img_files, coord):
    """
    Load freeSurfer outputs.

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


def do_kernel_smoothing(data, coord, bw_opt, log):
    """
    A wrap function for doing kernel smoothing.

    Parameters:
    ------------
    data (n, N): a np.array of imaging data
    bw_opt (1, ): a scalar of optimal bandwidth
    coord (N, dim): a np.array of coordinate 
    log: a logger

    Returns:
    ---------
    sm_data (n, N): smoothed and centered imaging data

    """
    ks = LocalLinear(data, coord)
    if not bw_opt:
        bw_list = ks.bw_cand()
        log.info(
            f"Selecting the optimal bandwidth from\n{np.round(bw_list, 3)}.")
        bw_opt = ks.gcv(bw_list, log)
    else:
        bw_opt = np.repeat(bw_opt, coord.shape[1])
    log.info(f"Doing kernel smoothing using the optimal bandwidth.\n")
    _, sm_data, _ = ks.smoother(bw_opt)
    if not isinstance(sm_data, np.ndarray):
        raise ValueError(
            'the bandwidth provided by --bw-opt may be problematic')
    sm_data = sm_data - np.mean(sm_data, axis=0)

    return sm_data


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
    if args.image_dir is None:
        raise ValueError('--image-dir is required')
    if args.image_suffix is None:
        raise ValueError('--image-suffix is required')
    if args.bw_opt is not None and args.bw_opt <= 0:
        raise ValueError('--bw-opt should be positive')

    args.image_dir = args.image_dir.split(',')
    args.image_suffix = args.image_suffix.split(',')
    if len(args.image_dir) != len(args.image_suffix):
        raise ValueError('--image-dir and --image-suffix do not match')
    for image_dir in args.image_dir:
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"{image_dir} does not exist")
    if args.surface_mesh is not None and not os.path.exists(args.surface_mesh):
        raise FileNotFoundError(f"{args.surface_mesh} does not exist")

    return args


def run(args, log):
    # check input
    args = check_input(args)

    # subjects to keep
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects are common in --keep.')
    else:
        keep_idvs = None

    # read images
    ids, img_files = get_image_list(
        args.image_dir, args.image_suffix, log, keep_idvs)
    if args.surface_mesh is not None:
        coord = nib.freesurfer.read_geometry(args.surface_mesh)[0]
        images = load_freesurfer(img_files, coord)
    else:
        images, coord = load_nifti(img_files)
    log.info(
        f"{images.shape[0]} subjects and {images.shape[1]} voxels are included in the imaging data.")

    # kernel smoothing
    log.info('Doing kernel smoothing using the local linear method ...')
    sm_images = do_kernel_smoothing(images, coord, args.bw_opt, log)

    # save images
    save_images(args.out + '_raw_images', images, coord, ids)
    save_images(args.out + '_sm_images', sm_images, coord, ids)
    log.info(f'Save the raw images to {args.out}_raw_images.h5')
    log.info(f'Save the smoothed images to {args.out}_raw_images.h5')
