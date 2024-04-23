import os
import logging
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from numpy.linalg import inv
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, hstack


class KernelSmooth:
    def __init__(self, data, coord):
        """
        Parameters:
        ------------
        ld_prefix: prefix of LD matrix file 
        data (n * N): raw imaging data
        coord (N * d): coordinates of points.

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
        x: vector or matrix in coordinate matrix
        bw (1 * 1): bandwidth 

        Returns:
        ---------
        Gaussian kernel function
       
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
        bw_list: a list of candidate bandwidths 
        log: a logger 

        Returns:
        ---------
        The optimal bandwidth
       
        """
        mse = np.zeros(len(bw_list))

        for cii, bw in enumerate(bw_list):
            log.info(f"Doing generalized cross-validation (GCV) for bandwidth {np.round(bw, 3)} ...")
            isvalid, y_sm, sparse_sm_weight = self.smoother(bw)
            if isvalid:
                # mse[cii] = np.mean(np.sum((self.data - y_sm) ** 2,
                #            axis=1) / (1 - np.sum(csc_matrix.diagonal(sparse_sm_weight)) / self.N) ** 2)
                # dis = np.sum((self.data - y_sm) ** 2, axis=1)
                mse[cii] = (np.mean(np.sum((self.data - y_sm) ** 2, axis=1)) / 
                            (1 - np.sum(sparse_sm_weight.diagonal()) / self.N + 10**-10) ** 2)
                if mse[cii] == 0:
                    mse[cii] = np.nan
                log.info(f"The MSE for bandwidth {np.round(bw, 3)} is {round(mse[cii], 3)}.")
            else:
                mse[cii] = np.Inf
        
        which_min = np.nanargmin(mse)
        if which_min == 0 or which_min == len(bw_list) - 1:
            log.info(("WARNING: the optimal bandwidth was obtained at the boundary, "
                      "which may not be the best one."))
        bw_opt = bw_list[which_min]
        min_mse = mse[which_min]
        log.info(f"The optimal bandwidth is {np.round(bw_opt, 3)} with MSE {round(min_mse, 3)}.")

        return bw_opt
    

    def bw_cand(self):
        """
        Generating a list of candidate bandwidths
    
        """
        bw_raw = self.N ** (-1 / (4 + self.d))
        weights = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
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
        bw (d * 1): bandwidth for d dimension
        
        Returns:
        ---------
        An Boolean variable that the bandwidth is valid to make the weight matrix sparse;
        the smooothed data, and the sparse weights.
        
        """
        sparse_sm_weight = dok_matrix((self.N, self.N), dtype=np.float32)
        for lii in range(self.N):
            t_mat0 = self.coord - self.coord[lii] # N * d
            t_mat = np.hstack((np.ones(self.N).reshape(-1, 1), t_mat0))
            dis = t_mat0 / bw
            close_points = (dis < 4) & (dis > -4)
            k_mat = csr_matrix((self._gau_kernel(dis[close_points]), np.where(close_points)), 
                               (self.N, self.d))
            k_mat = csc_matrix(np.prod(k_mat / bw, axis=1)) # can be faster
            k_mat_sparse = hstack([k_mat] * (self.d + 1))
            kx = k_mat_sparse.multiply(t_mat).T # (d+1) * N
            sm_weight = inv(kx @ t_mat + np.eye(self.d + 1) * 0.000001)[0, :] @ kx # N * 1
            large_weight_idxs = np.where(np.abs(sm_weight) > 1 / self.N)
            sparse_sm_weight[lii, large_weight_idxs] = sm_weight[large_weight_idxs]
        nonzero_weights = np.sum(sparse_sm_weight != 0, axis=0)
        if np.mean(nonzero_weights) > self.N // 10:
            self.logger.info((f"On average, the non-zero weight for each voxel are greater than {self.N // 10}. "
                              "Skip this bandwidth."))
            return False, None, None
        
        sm_data = self.data @ sparse_sm_weight.T
        return True, sm_data, sparse_sm_weight
    


def get_image_list(img_dirs, suffix, keep_idvs=None):
    img_files = {}
    
    for img_dir in img_dirs:
        for img_file in os.listdir(img_dir):
            img_id = img_file.replace(suffix, '')
            if (img_file.endswith(suffix) and ((keep_idvs is not None and img_id in keep_idvs) or
                (keep_idvs is None))):
                img_files[img_id] = os.path.join(img_dir, img_file)
    img_files = dict(sorted(img_files.items()))
    ids = pd.MultiIndex.from_arrays([img_files.keys(), img_files.keys()], names=['FID', 'IID'])
    
    return ids, list(img_files.values()) 



def load_nifti(img_files, log):
    try:
        img = nib.load(img_files[0])
    except:
        raise ValueError(('Cannot read the image, did you provide FreeSurfer images '
                          'but forget to provide a Freesurfer surface mesh?'))

    for i, img_file in enumerate(img_files):
        img = nib.load(img_file)
        data = img.get_fdata()
        if i == 0:
            idxs = data != 0 
            coord = np.stack(np.nonzero(data)).T
            n_voxels = np.sum(idxs)
            images = np.zeros((len(img_files), n_voxels), dtype=np.float32)
        images[i] = data[idxs]
        if i % 1000 == 0 and i > 0:
            log.info(f'Read {i+1} images.')

    return images, coord


def load_freesurfer(img_files, geometry, log):
    for i, img_file in enumerate(img_files):
        data = nib.freesurfer.read_morph_data(img_file)
        if i == 0:
            images = np.zeros((len(img_files), len(data)), dtype=np.float32)
        images[i] = data
        if i % 1000 == 0 and i > 0:
            log.info(f'Read {i+1} images.')
    coord = nib.freesurfer.read_geometry(geometry)[0]
    if coord.shape[0] != images.shape[1]:
        raise ValueError('The FreeSurfer geometry data and morphometry data have inconsistent coordinates.') 

    return images, coord


def do_kernel_smoothing(data, coord, bw_opt, log):
    """
    A wrap function for doing kernel smoothing

    Parameters:
    ------------
    data (n * N): an np.array of imaging data
    coord (N * d): an np.array of coordinate 

    Returns:
    ---------
    values: eigenvalues
    bases: eigenfunctions
    
    """
    ks = LocalLinear(data, coord)
    if not bw_opt:
        bw_list = ks.bw_cand()
        log.info(f"Selecting the optimal bandwidth from\n{np.round(bw_list, 3)}.")
        bw_opt = ks.gcv(bw_list, log)
    else:
        bw_opt = np.repeat(bw_opt, coord.shape[1])
    log.info(f"Doing kernel smoothing using the optimal bandwidth.\n")
    _, sm_data, _ = ks.smoother(bw_opt)
    if not isinstance(sm_data, np.ndarray):
        raise ValueError('The bandwidth provided by --bw-opt may be problematic.')
    
    return sm_data



def save_images(out, images, coord, id):
    with h5py.File(f'{out}.h5', 'w') as file:
        dset = file.create_dataset('images', data=images)
        file.create_dataset('id', data=np.array(id.tolist(), dtype='S10'))
        file.create_dataset('coord', data=coord)
        dset.attrs['id'] = 'id'
        dset.attrs['coord'] = 'coord'



def check_input(args):
    if args.image_dir is None:
        raise ValueError('--image-dir is required.')
    if args.image_suffix is None:
        raise ValueError('--image-suffix is required.')
    if args.bw_opt is not None and args.bw_opt <= 0:
        raise ValueError('--bw-opt should be positive.')

    args.image_dir = args.image_dir.split(',')
    for image_dir in args.image_dir:
        if not os.path.exists(image_dir):
            raise ValueError(f"{image_dir} does not exist.") 
    if args.keep is not None and not os.path.exists(args.keep):
        raise ValueError(f"{args.covar} does not exist.")
    if args.surface_mesh is not None and not os.path.exists(args.surface_mesh):
        raise ValueError(f"{args.surface_mesh} does not exist.")

    return args


def run(args, log):
    # check input
    args = check_input(args)
    
    # subjects to keep
    if args.keep:
        keep_idvs = pd.read_csv(args.keep, delim_whitespace=True, header=None, usecols=[0, 1],
                               dtype={0: str, 1: str})
        keep_idvs = keep_idvs.set_index(0).index # use only the first ID
        log.info(f'{len(keep_idvs)} subjects are in {args.keep}')
    else:
        keep_idvs = None

    # read images
    ids, img_files = get_image_list(args.image_dir, args.image_suffix, keep_idvs)
    log.info(f'Reading {len(ids)} images ...')
    if args.surface_mesh is not None:
        images, coord = load_freesurfer(img_files, args.surface_mesh, log)
    else:
        images, coord = load_nifti(img_files, log)
    log.info(f"{images.shape[0]} subjects and {images.shape[1]} voxels are included in the imaging data.")
    # np.savez(f"{args.out}_raw_images.npz", id = ids, coord = coord, images = images)
    
    # kernel smoothing
    log.info('Doing kernel smoothing using the local linear method ...')
    sm_images = do_kernel_smoothing(images, coord, args.bw_opt, log)
    # np.savez(f"{args.out}_sm_images.npz", id = ids, coord = coord, images = sm_data)
    
    # save images
    save_images(args.out + '_raw_images', images, coord, ids)
    save_images(args.out + '_sm_images', sm_images, coord, ids)
    log.info(f'Save the raw images to {args.out}_raw_images.h5')
    log.info(f'Save the smoothed images to {args.out}_raw_images.h5')