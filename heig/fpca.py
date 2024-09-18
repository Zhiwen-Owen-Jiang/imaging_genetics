import os
import logging
import h5py
import numpy as np
import pandas as pd
import concurrent.futures
import scipy.sparse as sp
from tqdm import tqdm
from functools import partial
from numpy.linalg import inv
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, hstack
from sklearn.decomposition import IncrementalPCA
from scipy.interpolate import make_interp_spline
import heig.input.dataset as ds


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
        gau_k = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

        return gau_k

    def smoother(self):
        raise NotImplementedError

    def gcv(self, bw_list, threads, temp_path, log):
        """
        Generalized cross-validation for selecting the optimal bandwidth

        Parameters:
        ------------
        bw_list: a array of candidate bandwidths
        threads: number of threads
        temp_path: temporay directory to save a sparse smoothing matrix
        log: a logger

        Returns:
        ---------
        sparse_sm_weight: the sparse smoothing matrix

        """
        score = np.zeros(len(bw_list), dtype=np.float32)
        min_score = np.Inf

        for cii, bw in enumerate(bw_list):
            log.info(
                f"Doing generalized cross-validation (GCV) for bandwidth {np.round(bw, 3)} ..."
            )
            sparse_sm_weight = self.smoother(bw, threads)
            if sparse_sm_weight is not None:
                mean_sm_weight_diag = np.sum(sparse_sm_weight.diagonal()) / self.N
                mean_diff = self._calculate_diff_parallel(sparse_sm_weight, threads)
                score[cii] = mean_diff / (1 - mean_sm_weight_diag + 10**-10) ** 2

                if score[cii] == 0:
                    score[cii] = np.nan
                    log.info(f"This bandwidth is invalid.")
                if score[cii] < min_score:
                    min_score = score[cii]
                    self._save_sparse_sm_weight(sparse_sm_weight, temp_path)
                log.info(
                    f"The GCV score for bandwidth {np.round(bw, 3)} is {score[cii]:.3f}."
                )
            else:
                score[cii] = np.Inf

        which_min = np.nanargmin(score)
        if which_min == 0 or which_min == len(bw_list) - 1:
            log.info(
                (
                    "WARNING: the optimal bandwidth was obtained at the boundary, "
                    "which may not be the best one."
                )
            )
        bw_opt = bw_list[which_min]
        min_mse = score[which_min]
        log.info(
            f"The optimal bandwidth is {np.round(bw_opt, 3)} with GCV score {min_mse:.3f}."
        )

        sparse_sm_weight = self._load_sparse_sm_weight(temp_path)

        return sparse_sm_weight

    @staticmethod
    def _calculate_diff(images_, sparse_sm_weight):
        return np.sum((images_ - images_ @ sparse_sm_weight.T) ** 2)

    def _calculate_diff_parallel(self, sparse_sm_weight, threads):
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(self._calculate_diff, images_, sparse_sm_weight)
                for images_ in image_reader(self.images, self.id_idxs)
            ]
            diff = [future.result() for future in futures]
        mean_diff = np.sum(diff) / self.n

        return mean_diff

    @staticmethod
    def _save_sparse_sm_weight(sparse_sm_weight, temp_path):
        sparse_sm_weight = sparse_sm_weight.tocoo()
        sp.save_npz(f"{temp_path}.npz", sparse_sm_weight)

    @staticmethod
    def _load_sparse_sm_weight(temp_path):
        sparse_sm_weight = sp.load_npz(f"{temp_path}.npz")
        sparse_sm_weight = sparse_sm_weight.todok()
        return sparse_sm_weight

    def bw_cand(self):
        """
        Generating a array of candidate bandwidths

        Returns:
        ---------
        bw_list (6, dim): candidate bandwidth

        """
        bw_raw = self.N ** (-1 / (4 + self.d))
        # weights = [0.2, 0.5, 1, 2, 5, 10]
        weights = [0.5, 1, 2, 3, 5, 10]
        bw_list = np.zeros((len(weights), self.d), dtype=np.float32)

        for i, weight in enumerate(weights):
            bw_list[i, :] = np.repeat(weight * bw_raw, self.d)

        return bw_list


class LocalLinear(KernelSmooth):
    def __init__(self, images, coord, id_idxs):
        super().__init__(images, coord, id_idxs)
        self.logger = logging.getLogger(__name__)

    def smoother(self, bw, threads):
        """
        Local linear smoother

        Parameters:
        ------------
        bw (dim, 1): bandwidth for dim dimension
        threads: number of threads

        Returns:
        ---------
        sparse_sm_weight (N, N): sparse kernel smoothing weights or None

        """
        sparse_sm_weight = dok_matrix((self.N, self.N), dtype=np.float32)

        partial_function = partial(self._sm_weight, bw)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(partial_function, idx): idx for idx in range(self.N)
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                sm_weight, large_weight_idxs = future.result()
                sparse_sm_weight[idx, large_weight_idxs] = sm_weight

        nonzero_weights = np.sum(sparse_sm_weight != 0, axis=0)
        if np.mean(nonzero_weights) > self.N // 10:
            self.logger.info(
                (
                    f"On average, the non-zero weight for each voxel "
                    f"are greater than {self.N // 10}. "
                    "Skip this bandwidth."
                )
            )
            return None

        return sparse_sm_weight

    def _sm_weight(self, bw, idx):
        """
        Computing smoothing weight for a voxel

        Parameters:
        ------------
        bw (dim, 1): bandwidth for dim dimension
        idx: voxel index

        """
        t_mat0 = self.coord - self.coord[idx]  # N * d
        t_mat = np.hstack((np.ones(self.N).reshape(-1, 1), t_mat0))
        dis = t_mat0 / bw
        close_points = (dis < 4) & (dis > -4)  # keep only nearby voxels
        k_mat = csr_matrix(
            (self._gau_kernel(dis[close_points]), np.where(close_points)),
            (self.N, self.d),
        )
        k_mat = csc_matrix(
            np.prod((k_mat / bw).toarray(), axis=1)
        ).T  # can be faster, update for scipy 1.11
        k_mat_sparse = hstack([k_mat] * (self.d + 1))
        kx = k_mat_sparse.multiply(t_mat).T  # (d+1) * N
        sm_weight = inv(kx @ t_mat + np.eye(self.d + 1) * 0.000001)[0, :] @ kx  # N * 1
        large_weight_idxs = np.where(np.abs(sm_weight) > 1 / self.N)

        return sm_weight[large_weight_idxs], large_weight_idxs


def image_reader(images, id_idxs):
    """
    Reading imaging data in chunks, each chunk is ~5 GB

    Parameters:
    ------------
    images (n, N): raw imaging data reference
    id_idxs (n1, ): numerical indices of subjects that included in the analysis

    Returns:
    ---------
    A generator of images

    """
    N = images.shape[1]
    n = len(id_idxs)
    memory_use = n * N * np.dtype(np.float32).itemsize / (1024**3)
    if memory_use <= 5:
        batch_size = n
    else:
        batch_size = int(n / memory_use * 5)

    for i in range(0, n, batch_size):
        id_idx_chuck = id_idxs[i : i + batch_size]
        yield images[id_idx_chuck]


def do_kernel_smoothing(
    raw_image_dir, sm_image_dir, keep_idvs, bw_opt, threads, temp_path, log
):
    """
    A wrapper function for doing kernel smoothing.

    Parameters:
    ------------
    raw_image_dir: directory to HDF5 file of raw images
    sm_image_dir: directory to HDF5 file of smoothed images
    keep_idvs: pd.MultiIndex of subjects to keep
    bw_opt (1, ): a scalar of optimal bandwidth
    threads: number of threads
    temp_path: temporay directory to save a sparse smoothing matrix
    log: a logger

    Returns:
    ---------
    subject_wise_mean (N, ): sample mean of smoothed images, used in PCA

    """
    with h5py.File(raw_image_dir, "r") as file:
        images = file["images"]
        coord = file["coord"][:]
        ids = file["id"][:]
        ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])

        log.info(
            f"{len(ids)} subjects and {coord.shape[0]} voxels (vertices) read from {raw_image_dir}"
        )

        if keep_idvs is not None:
            common_ids = ds.get_common_idxs(ids, keep_idvs)
        else:
            common_ids = ids
        id_idxs = np.arange(len(ids))[ids.isin(common_ids)]
        log.info(f"Using {len(id_idxs)} subjects.")

        ks = LocalLinear(images, coord, id_idxs)
        if bw_opt is None:
            log.info("\nDoing kernel smoothing ...")
            bw_list = ks.bw_cand()
            log.info(f"Selecting the optimal bandwidth from\n{np.round(bw_list, 3)}.")
            sparse_sm_weight = ks.gcv(bw_list, threads, temp_path, log)
        else:
            bw_opt = np.repeat(bw_opt, coord.shape[1])
            log.info(f"Doing kernel smoothing using the optimal bandwidth.")
            sparse_sm_weight = ks.smoother(bw_opt, threads)

        n_voxels = images.shape[1]
        n_subjects = len(id_idxs)
        if sparse_sm_weight is not None:
            subject_wise_mean = np.zeros(n_voxels, dtype=np.float32)
            with h5py.File(sm_image_dir, "w") as h5f:
                sm_images = h5f.create_dataset(
                    "sm_images", shape=(n_subjects, n_voxels), dtype="float32"
                )
                start_idx, end_idx = 0, 0
                for images_ in image_reader(images, id_idxs):
                    start_idx = end_idx
                    end_idx += images_.shape[0]
                    sm_image_ = images_ @ sparse_sm_weight.T
                    sm_images[start_idx:end_idx] = sm_image_
                    subject_wise_mean += np.sum(sm_image_, axis=0)
                subject_wise_mean /= n_subjects
                h5f.create_dataset(
                    "id", data=np.array(common_ids.tolist(), dtype="S10")
                )
                h5f.create_dataset("coord", data=coord)
                sm_images.attrs["id"] = "id"
                sm_images.attrs["coord"] = "coord"
        else:
            raise ValueError("the bandwidth provided by --bw-opt may be problematic")

    return subject_wise_mean


class FPCA:
    def __init__(self, n_sub, n_voxels, compute_all, n_ldrs):
        """
        Parameters:
        ------------
        n_sub: sample size
        n_voxels: the number of voxels
        compute_all: a boolean variable for computing all components
        n_ldrs: a specified number of components

        """
        max_n_pc = np.min((n_sub, n_voxels))
        self.logger = logging.getLogger(__name__)
        self.n_top = self._get_n_top(n_ldrs, max_n_pc, compute_all)
        self.batch_size = self._get_batch_size(max_n_pc, n_sub)
        self.n_batches = n_sub // self.batch_size
        self.ipca = IncrementalPCA(n_components=self.n_top, batch_size=self.batch_size)
        self.logger.info(f"Computing the top {self.n_top} components.")

    def _get_n_top(self, n_ldrs, max_n_pc, compute_all):
        """
        Determine the number of top components to compute in PCA.

        Parameters:
        ------------
        n_ldrs: a specified number of components
        max_n_pc: the maximum possible number of components
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
                self.logger.info(
                    "WARNING: --n-ldrs is greater than the maximum #components."
                )
            else:
                n_top = n_ldrs
                if n_ldrs < int(max_n_pc / 5):
                    self.logger.info(
                        (
                            "WARNING: --n-ldrs is less than 20% of the maximum #components. "
                            "The number of LDRs for a proportion of variance and "
                            "the effective number of indenpendent voxels may be downward biased."
                        )
                    )
        else:
            n_top = int(max_n_pc / 5)

        return n_top

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
                batch_size = n_sub
            else:
                batch_size = n_sub // (n_sub // 50000 + 1)
        else:
            if self.n_top > 15000 or n_sub > 50000:
                i = 2
                while n_sub // i > 50000:
                    i += 1
                batch_size = n_sub // i
            else:
                batch_size = n_sub

        return np.max((batch_size, self.n_top))


def do_fpca(sm_image_dir, subject_wise_mean, args, log):
    """
    A wrapper function for doing functional PCA.

    Parameters:
    ------------
    sm_image_dir: directory to HDF5 file of smoothed images
    subject_wise_mean (N, ): sample mean of smoothed images, used in PCA
    args: arguments
    log: a logger

    Returns:
    ---------
    values (n_top, ): eigenvalues
    bases (N, n_top): functional bases
    fpca.n_top (1, ): #PCs

    """
    with h5py.File(sm_image_dir, "r") as file:
        sm_images = file["sm_images"]
        n_subjects, n_voxels = sm_images.shape

        # setup parameters
        log.info(f"\nDoing functional PCA ...")
        fpca = FPCA(n_subjects, n_voxels, args.all_pc, args.n_ldrs)

        # incremental PCA
        max_avail_n_sub = fpca.n_batches * fpca.batch_size
        log.info(
            (
                f"The smoothed images are split into {fpca.n_batches} batch(es), "
                f"with batch size {fpca.batch_size}."
            )
        )
        for i in tqdm(
            range(0, max_avail_n_sub, fpca.batch_size),
            desc=f"{fpca.n_batches} batch(es)",
        ):
            fpca.ipca.partial_fit(
                sm_images[i : i + fpca.batch_size] - subject_wise_mean
            )
        values = (fpca.ipca.singular_values_**2).astype(np.float32)
        bases = fpca.ipca.components_.T
        bases = bases.astype(np.float32)

    return values, bases, fpca.n_top


class EigenValues:
    """
    Predicting uncomputed eigenvalues using a B-spline

    """

    def __init__(self, values, max_n_pc):
        """
        Parameters:
        ------------
        values (n_top, ): eigenvalues
        max_n_pc (1, ): maximum #pc

        """
        self.values = values
        self.max_n_pc = max_n_pc
        self.logger = logging.getLogger(__name__)

        self.imputed_values = self._bspline()
        self.eff_num = self._eff_num()
        self.prop_ldrs_df = self._print_prop_ldr()

    def _bspline(self):
        """
        Using a B-spline with degree of 1 to predict log-eigenvalues

        """
        self.logger.info("Imputing uncomputed eigenvalues using a B-spline (degree=1).")
        n_values = len(self.values)
        x_train = np.arange(n_values)
        y_train = np.log(self.values)
        spline = make_interp_spline(x_train, y_train, k=1)
        x_pred = np.arange(n_values, self.max_n_pc)
        y_pred = spline(x_pred)
        imputed_values = np.concatenate([self.values, np.exp(y_pred)]).astype(
            np.float32
        )

        return imputed_values

    def _eff_num(self):
        """
        Computing effective number of independent voxels

        """
        norm_values = self.imputed_values / self.imputed_values[0]
        eff_num = np.sum(norm_values) ** 2 / np.sum((norm_values) ** 2)

        return eff_num

    def _print_prop_ldr(self):
        """
        Computing the number of LDRs required for varying proportions of variance

        """
        prop_var = np.cumsum(self.imputed_values) / np.sum(self.imputed_values)
        prop_ldrs = {}
        for prop in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            prop_ldrs[prop] = np.sum(prop_var <= prop) + 1

        max_key_len = max(len(str(key)) for key in prop_ldrs.keys())
        max_val_len = max(len(str(value)) for value in prop_ldrs.values())
        max_len = max([max_key_len, max_val_len])
        keys_str = "  ".join(f"{str(key):<{max_len}}" for key in prop_ldrs.keys())
        values_str = "  ".join(
            f"{str(value):<{max_len}}" for value in prop_ldrs.values()
        )

        self.logger.info(
            "The number of LDRs for preserving varying proportions of image variance:"
        )
        self.logger.info(keys_str)
        self.logger.info(values_str)

        prop_ldrs_df = pd.DataFrame.from_dict(prop_ldrs, orient="index")
        prop_ldrs_df.index.name = "prop_var"
        prop_ldrs_df = prop_ldrs_df.rename({0: "n_ldrs"}, axis=1)

        return prop_ldrs_df


def check_input(args, log):
    if args.image is None:
        raise ValueError("--image is required")
    if args.all_pc:
        log.info(
            (
                "WARNING: computing all principal components might be very time "
                "and memory consuming when images are of high resolution."
            )
        )
    if args.all_pc and args.n_ldrs is not None:
        log.info("--all-pc is ignored as --n-ldrs specified.")
        args.all_pc = False
    if args.bw_opt is not None and args.bw_opt <= 0:
        raise ValueError("--bw-opt should be positive")

    temp_path = os.path.join(os.path.dirname(args.out), "temp_sparse_sm_weight")
    i = 0
    while os.path.exists(temp_path + str(i)):
        i += 1
    temp_path += str(i)

    return temp_path


def run(args, log):
    # check input
    temp_path = check_input(args, log)

    try:
        # kernel smoothing
        sm_image_dir = f"{args.out}_sm_images.h5"
        subject_wise_mean = do_kernel_smoothing(
            args.image,
            sm_image_dir,
            args.keep,
            args.bw_opt,
            args.threads,
            temp_path,
            log,
        )

        # fPCA
        values, bases, n_top = do_fpca(sm_image_dir, subject_wise_mean, args, log)
        eigenvalues = EigenValues(values, bases.shape[0])

        np.save(f"{args.out}_bases_top{n_top}.npy", bases)
        np.save(f"{args.out}_eigenvalues.npy", eigenvalues.imputed_values)
        eigenvalues.prop_ldrs_df.to_csv(f"{args.out}_ldrs_prop_var.txt", sep="\t")
        log.info(
            (
                f"The effective number of independent voxels (vertices) is {eigenvalues.eff_num:.3f}, "
                f"which can be used in the Bonferroni p-value threshold (e.g., 0.05/{eigenvalues.eff_num:.3f}) "
                "across all voxels (vertices).\n"
            )
        )
        log.info(f"Save the top {n_top} bases to {args.out}_bases_top{n_top}.npy")
        log.info(f"Save the eigenvalues to {args.out}_eigenvalues.npy")
        log.info(f"Save the number of LDRs table to {args.out}_ldrs_prop_var.txt")

    finally:
        if os.path.exists(f"{temp_path}.npz"):
            os.remove(f"{temp_path}.npz")
        if os.path.exists(sm_image_dir):
            os.remove(sm_image_dir)
