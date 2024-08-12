import os
import logging
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.linalg import inv
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, hstack
from sklearn.decomposition import IncrementalPCA
import heig.input.dataset as ds

"""
TODO: parallel

"""


class KernelSmooth:
    def __init__(self, images, coord, id_idxs):
        """
        Parameters:
        ------------
        images (n, N): raw imaging data reference
        coord (N, dim): coordinates
        id_idxs (n1, ): numerical indices of subjects that included in the analysis

        """
        self.images = images
        self.coord = coord
        self.id_idxs = id_idxs
        self.n = len(id_idxs)
        self.N, self.d = self.coord.shape

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
            log.info(f"Doing generalized cross-validation (GCV) for bandwidth {np.round(bw, 3)} ...")
            sparse_sm_weight = self.smoother(bw)
            mean_sm_weight_diag = np.sum(sparse_sm_weight.diagonal()) / self.N
            if sparse_sm_weight is not None:
                diff = [np.sum((self.images[id_idx] - self.images[id_idx] @ sparse_sm_weight) ** 2) for id_idx in self.id_idxs]
                mean_diff = np.mean(diff)
                score[cii] = mean_diff / (1 - mean_sm_weight_diag + 10**-10) ** 2
                if score[cii] == 0:
                    score[cii] = np.nan
                    log.info(f'This bandwidth is invalid.')
                log.info(f"The GCV score for bandwidth {np.round(bw, 3)} is {round(score[cii], 3)}.")
            else:
                score[cii] = np.Inf

        which_min = np.nanargmin(score)
        if which_min == 0 or which_min == len(bw_list) - 1:
            log.info(("WARNING: the optimal bandwidth was obtained at the boundary, "
                      "which may not be the best one."))
        bw_opt = bw_list[which_min]
        min_mse = score[which_min]
        log.info(f"The optimal bandwidth is {np.round(bw_opt, 3)} with GCV score {round(min_mse, 3)}.")

        return bw_opt

    def bw_cand(self):
        """
        Generating a array of candidate bandwidths

        Returns:
        ---------
        bw_list (6, dim): candidate bandwidth

        """
        bw_raw = self.N ** (-1 / (4 + self.d))
        weights = [0.2, 0.5, 1, 2, 5, 10]
        bw_list = np.zeros((len(weights), self.d))

        for i, weight in enumerate(weights):
            bw_list[i, :] = np.repeat(weight * bw_raw, self.d)

        return bw_list


class LocalLinear(KernelSmooth):
    def __init__(self, images, coord, id_idxs):
        super().__init__(images, coord, id_idxs)
        self.logger = logging.getLogger(__name__)

    def smoother(self, bw):
        """
        Local linear smoother. 

        Parameters:
        ------------
        bw (dim, 1): bandwidth for dim dimension

        Returns:
        ---------
        sparse_sm_weight (N, N): sparse kernel smoothing weights or None

        """
        sparse_sm_weight = dok_matrix((self.N, self.N), dtype=np.float32)
        for lii in range(self.N):
            t_mat0 = self.coord - self.coord[lii]  # N * d
            t_mat = np.hstack((np.ones(self.N).reshape(-1, 1), t_mat0))
            dis = t_mat0 / bw
            close_points = (dis < 4) & (dis > -4)  # keep only nearby voxels
            k_mat = csr_matrix((self._gau_kernel(dis[close_points]), np.where(close_points)),
                               (self.N, self.d))
            k_mat = csc_matrix(np.prod((k_mat / bw).toarray(), axis=1)).T  # can be faster, update for scipy 1.11
            k_mat_sparse = hstack([k_mat] * (self.d + 1))
            kx = k_mat_sparse.multiply(t_mat).T  # (d+1) * N
            sm_weight = inv(kx @ t_mat + np.eye(self.d + 1) * 0.000001)[0, :] @ kx  # N * 1
            large_weight_idxs = np.where(np.abs(sm_weight) > 1 / self.N)
            sparse_sm_weight[lii,large_weight_idxs] = sm_weight[large_weight_idxs]
        nonzero_weights = np.sum(sparse_sm_weight != 0, axis=0)
        if np.mean(nonzero_weights) > self.N // 10:
            self.logger.info((f"On average, the non-zero weight for each voxel "
                              f"are greater than {self.N // 10}. "
                              "Skip this bandwidth."))
            return None
        
        return sparse_sm_weight


def do_kernel_smoothing(raw_image_dir, sm_image_dir, keep_idvs, bw_opt, log):
    """
    A wrap function for doing kernel smoothing.

    Parameters:
    ------------
    raw_image_dir: directory to HDF5 file of raw images
    sm_image_dir: directory to HDF5 file of smoothed images
    keep_idvs: pd.MultiIndex of subjects to keep
    bw_opt (1, ): a scalar of optimal bandwidth
    log: a logger

    Returns:
    ---------
    subject_wise_mean (N, ): sample mean of smoothed images, used in PCA

    """
    with h5py.File(raw_image_dir, 'r') as file:
        images = file['images']
        coord = file['coord'][:]
        ids = file['id'][:]
        ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=['FID', 'IID'])

        if keep_idvs is not None:
            common_ids = ds.get_common_idxs(ids, keep_idvs)
        else:
            common_ids = ids
        id_idxs = np.arange(len(ids))[ids.isin(common_ids)]

        ks = LocalLinear(images, coord, id_idxs)
        if not bw_opt:
            bw_list = ks.bw_cand()
            log.info(f"Selecting the optimal bandwidth from\n{np.round(bw_list, 3)}.")
            bw_opt = ks.gcv(bw_list, log)
        else:
            bw_opt = np.repeat(bw_opt, coord.shape[1])
        log.info(f"Doing kernel smoothing using the optimal bandwidth.\n")
        sparse_sm_weight = ks.smoother(bw_opt)

        n_voxels = images.shape[1]
        n_subjects = len(id_idxs)
        if sparse_sm_weight is not None:
            subject_wise_mean = np.zeros(n_voxels)
            with h5py.File(sm_image_dir, 'w') as h5f:
                sm_images = h5f.create_dataset('sm_images', shape=(n_subjects, n_voxels), dtype='float32')
                for i in range(n_subjects):
                    sm_image_i = images[i] @ sparse_sm_weight
                    sm_images[i] = sm_image_i
                    subject_wise_mean += sm_image_i / n_subjects
                h5f.create_dataset('id', data=np.array(common_ids.tolist(), dtype='S10'))
                h5f.create_dataset('coord', data=coord)
                images.attrs['id'] = 'id'
                images.attrs['coord'] = 'coord'
        else:
            raise ValueError('the bandwidth provided by --bw-opt may be problematic')

    return subject_wise_mean


class fPCA:
    def __init__(self, n_sub, max_n_pc, dim, compute_all, n_ldrs):
        """
        Parameters:
        ------------
        n_sub: the sample size
        max_n_pc: the maximum possible number of components
        dim: the dimension of images
        compute_all: a boolean variable for computing all components
        n_ldrs: a specified number of components
        
        """
        self.logger = logging.getLogger(__name__)
        self.n_top = self._get_n_top(n_ldrs, max_n_pc, n_sub, dim, compute_all)
        self.batch_size = self._get_batch_size(max_n_pc, n_sub)
        self.n_batches = n_sub // self.batch_size
        self.ipca = IncrementalPCA(n_components=self.n_top, batch_size=self.batch_size)
        self.logger.info(f"Computing the top {self.n_top} components.")

    def _get_batch_size(self, max_n_pc, n_sub):
        """
        Adaptively determine batch size

        Parameters:
        ------------
        max_n_pc: the maximum possible number of components
        n_sub: the sample size

        Returns:
        ---------
        batch size for IncrementalPCA

        """
        if max_n_pc <= 15000:
            if n_sub <= 50000:
                return n_sub
            else:
                return n_sub // (n_sub // 50000 + 1)
        else:
            if self.n_top > 15000 or n_sub > 50000:
                i = 2
                while n_sub // i > 50000:
                    i += 1
                return n_sub // i
            else:
                return n_sub

    def _get_n_top(self, n_ldrs, max_n_pc, n_sub, dim, compute_all):
        """
        Determine the number of top components to compute in PCA.

        Parameters:
        ------------
        n_ldrs: a specified number of components
        max_n_pc: the maximum possible number of components
        n_sub: the sample size
        dim: the dimension of images
        compute_all: a boolean variable for computing all components

        Returns:
        ---------
        n_top: the number of top components to compute in PCA

        """
        if compute_all:
            n_top = max_n_pc
        elif n_ldrs is not None:
            if n_ldrs > max_n_pc:
                n_top = max_n_pc
                self.logger.info('WARNING: --n-ldrs is greater than the maximum #components.')
            else:
                n_top = n_ldrs
        else:
            if dim == 1:
                n_top = max_n_pc
            else:
                n_top = int(max_n_pc / (dim - 1))

        n_top = np.min((n_top, n_sub))
        return n_top


def do_fpca(sm_image_dir, subject_wise_mean, args, log):
    with h5py.File(sm_image_dir, 'r') as file:
        sm_images = file['images']
        n_subjects, n_voxels = sm_images.shape
        coord = file['coord']
        _, dim = coord.shape

        # setup parameters
        log.info(f'Doing functional PCA ...')
        max_n_pc = np.min((n_subjects, n_voxels))
        fpca = fPCA(n_subjects, max_n_pc, dim, args.all, args.n_ldrs)

        # incremental PCA
        max_avail_n_sub = fpca.n_batches * fpca.batch_size
        log.info((f'The smoothed images are split into {fpca.n_batches} batch(es), '
                    f'with batch size {fpca.batch_size}.'))
        for i in tqdm(range(0, max_avail_n_sub, fpca.batch_size), desc=f"{fpca.n_batches} batch(es)"):
            fpca.ipca.partial_fit(sm_images[i: i+fpca.batch_size] - subject_wise_mean)
        values = fpca.ipca.singular_values_ ** 2
        bases = fpca.ipca.components_.T
        eff_num = np.sum(values) ** 2 / np.sum(values ** 2)

    return values, bases, eff_num, fpca.n_top


def determine_n_ldr(values, prop, log):
    """
    Determine the number of LDRs for preserving a proportion of variance

    Parameters:
    ------------
    values: a np.array of eigenvalues
    prop: a scalar of proportion between 0 and 1
    log: a logger

    Returns:
    ---------
    n_opt: the number of LDRs

    """
    eff_num = np.sum(values) ** 2 / np.sum(values ** 2)
    prop_var = np.cumsum(values) / np.sum(values)
    idxs = (prop_var <= prop) & (values != 0)
    n_idxs = np.sum(idxs) + 1
    n_opt = max(n_idxs, int(eff_num) + 1)
    var_prop = np.sum(values[:n_opt]) / np.sum(values)
    log.info((f'Approximately {round(var_prop * 100, 1)}% variance '
            f'is captured by the top {n_opt} components.\n'))
    return n_opt


def check_input(args, log):
    if args.image is None:
        raise ValueError('--image is required')
    if args.all:
        log.info(('WARNING: computing all principal components might be very time '
                  'and memory consuming when images are huge.'))
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0')
    if args.all and args.n_ldrs is not None:
        log.info('--all is ignored as --n-ldrs specified.')
        args.all = False
    if args.prop is not None:
        if args.prop <= 0 or args.prop > 1:
            raise ValueError('--prop should be between 0 and 1')
        elif args.prop < 0.8:
            log.info('WARNING: keeping less than 80% of variance will have bad performance.')
    if args.bw_opt is not None and args.bw_opt <= 0:
        raise ValueError('--bw-opt should be positive')

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"{args.image} does not exist")


def run(args, log):
    # check input
    check_input(args, log)

    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None

    # kernel smoothing
    sm_image_dir = f'{args.out}_sm_images.h5'
    subject_wise_mean = do_kernel_smoothing(args.image, sm_image_dir, 
                                            args.bw_opt, keep_idvs, log)
    log.info(f'Save smoothed images to {sm_image_dir}')

    # fPCA
    values, bases, eff_num, n_top = do_fpca(sm_image_dir, subject_wise_mean, args, log)

    # keep components
    if args.prop:
        n_opt = determine_n_ldr(values, args.prop, log)
    else:
        n_opt = n_top
    
    np.save(f"{args.out}_bases_top{n_opt}.npy", bases)
    np.save(f"{args.out}_eigenvalues_top{n_top}.npy", values)
    log.info((f"The effective number of independent voxels is {round(eff_num, 3)}, "
              f"which can be used in the Bonferroni p-value threshold (e.g., 0.05/{round(eff_num, 3)}) "
              "across all voxels."))
    log.info(f"Save the top {n_opt} bases to {args.out}_bases_top{n_opt}.npy")
    log.info(f"Save the top {n_top} eigenvalues to {args.out}_eigenvalues_top{n_top}.npy")