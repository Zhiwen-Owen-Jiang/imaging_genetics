import sys
import os
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import heig.input.dataset as ds
from heig import utils


class Dataset():
    def __init__(self, dir):
        """
        Parameters:
        ------------
        dir: diretory to the dataset

        """
        if dir.endswith('dat'):
            self.data = pickle.load(open(dir, 'rb'))
            if not 'FID' in self.data.columns or not 'IID' in self.data.columns:
                raise ValueError('FID and IID do not exist.')
        else:
            openfunc, compression = utils.check_compression(dir)
            self._check_header(openfunc, compression, dir)
            self.data = pd.read_csv(dir, delim_whitespace=True, compression=compression,
                                    na_values=[-9, 'NONE', '.'], dtype={'FID': str, 'IID': str})
        if self.data[['FID', 'IID']].duplicated().any():
            first_dup = self.data.loc[self.data[['FID', 'IID']].duplicated(), [
                'FID', 'IID']]
            raise ValueError(f'Subject {list(first_dup)} is duplicated.')
        if len(self.data.columns) != len(set(self.data.columns)):
            raise ValueError('Duplicated columns are not allowed.')
        self.data = self.data.set_index(['FID', 'IID'])
        self.data = self.data.sort_index()
        self.logger = logging.getLogger(__name__)
        self._remove_na_inf()

    def _check_header(self, openfunc, compression, dir):
        """
        The dataset should have a header: FID, IID, ...

        Parameters:
        ------------
        openfunc: function to read the first line
        dir: diretory to the dataset

        """
        header = openfunc(dir).readline().split()
        if len(header) == 1:
            raise ValueError(
                'Only one column detected, check your input file and delimiter.')
        if compression is not None:
            header[0] = str(header[0], 'UTF-8')
            header[1] = str(header[1], 'UTF-8')
        if header[0] != 'FID' or header[1] != 'IID':
            raise ValueError('The first two column names must be FID and IID.')

    def _remove_na_inf(self):
        """
        Removing rows with any missing values

        """
        bad_idxs = (self.data.isin([np.inf, -np.inf, np.nan])).any(axis=1)
        self.data = self.data.loc[~bad_idxs]
        self.logger.info(
            f"{sum(bad_idxs)} row(s) with missing or infinite values were removed.")

    def keep(self, idx):
        """
        Extracting rows using indices
        Sorting the data 

        Parameters:
        ------------
        idx: indices of the data

        """
        self.data = self.data.loc[idx]


class Covar(Dataset):
    def __init__(self, dir, cat_covar_list=None):
        """
        Parameters:
        ------------
        dir: diretory to the dataset

        """
        super().__init__(dir)
        self.cat_covar_list = cat_covar_list

    def cat_covar_intercept(self):
        """
        Convert categorical covariates to dummy variables,
        and add the intercept.

        """
        if self.cat_covar_list is not None:
            catlist = self.cat_covar_list.split(',')
            self.logger.info(
                f"{len(catlist)} categorical variables provided by --cat-covar-list.")
            self._check_validcatlist(catlist)
            self._dummy_covar(catlist)
        self._add_intercept()
        if self._check_singularity():
            raise ValueError('The covarite matrix is singular.')

    def _check_validcatlist(self, catlist):
        """
        Checking if all categorical covariates exist

        Parameters:
        ------------
        catlist: a list of categorical covariates

        """
        for cat in catlist:
            if cat not in self.data.columns:
                raise ValueError(f"{cat} cannot be found in the covariates.")

    def _dummy_covar(self, catlist):
        """
        Converting categorical covariates to dummy variables

        Parameters:
        ------------
        catlist: a list of categorical covariates

        """
        covar_df = self.data[catlist]
        qcovar_df = self.data[self.data.columns.difference(catlist)]
        _, q = covar_df.shape
        for i in range(q):
            covar_dummy = pd.get_dummies(covar_df.iloc[:, 0], drop_first=True)
            if len(covar_dummy) == 0:
                raise ValueError(f"{covar_df.columns[i]} have only one level.")
            covar_df = pd.concat([covar_df, covar_dummy], axis=1)
            covar_df.drop(covar_df.columns[0], inplace=True, axis=1)
        self.data = pd.concat([covar_df, qcovar_df], axis=1)

    def _add_intercept(self):
        """
        Adding the intercept

        """
        n = self.data.shape[0]
        const = pd.Series(np.ones(n), index=self.data.index, dtype=int)
        self.data = pd.concat([const, self.data], axis=1)

    def _check_singularity(self):
        """
        Checking if a matrix is singular
        True means singular

        """
        if len(self.data.shape) == 1:
            return self.data == 0
        else:
            return np.linalg.cond(self.data) >= 1/sys.float_info.epsilon

    def keep(self, idx):
        """
        Extracting rows using indices
        Sorting the data 

        Parameters:
        ------------
        idx: indices of the data

        """
        self.data = self.data.loc[idx]


def projection_ldr(ldr, covar):
    """
    Computing S'(I - M)S = S'S - S'X(X'X)^{-1}X'S,
    where I is the identity matrix, M is the project matrix for X,
    S is the LDR matrix

    Parameters:
    ------------
    ldr (N * r): low-dimension representaion of imaging data
    covar (n * p): covariates, including the intercept

    Returns:
    ---------
    Projected inner product of LDR

    """
    inner_ldr = np.dot(ldr.T, ldr)
    inner_covar = np.dot(covar.T, covar)
    # nonsingularity has been checked
    inner_covar_inv = np.linalg.inv(inner_covar)
    ldr_covar = np.dot(ldr.T, covar)
    part2 = np.dot(np.dot(ldr_covar, inner_covar_inv), ldr_covar.T)
    # part3 = np.dot(np.dot(covar, inner_covar_inv), ldr_covar.T)

    # return ldr - part3, inner_ldr - part2
    return inner_ldr - part2


def determine_n_ldr(values, prop, log):
    eff_num = np.sum(values) ** 2 / np.sum(values ** 2)
    prop_var = np.cumsum(values) / np.sum(values)
    idxs = (prop_var <= prop) & (values != 0)
    n_idxs = np.sum(idxs) + 1
    n_opt = max(n_idxs, int(eff_num) + 1)
    var_prop = np.sum(values[:n_opt]) / np.sum(values)
    log.info(
        f'Approximately {round(var_prop * 100, 1)}% variance is captured by the top {n_opt} components.\n')
    return n_opt


def check_input(args, log):
    if args.image is None:
        raise ValueError('--image is required.')
    if args.sm_image is None:
        raise ValueError('--sm-image is required.')
    if args.covar is None:
        raise ValueError('--covar is required.')
    if args.all:
        log.info(('WARNING: computing all principal components might be very time '
                  'and memory consuming when images are huge.'))
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0.')
    if args.all and args.n_ldrs is not None:
        log.info('--all is ignored as --n-ldrs specified.')
        args.all = False
    if args.prop is not None:
        if args.prop <= 0 or args.prop > 1:
            raise ValueError('--prop should be between 0 and 1.')
        elif args.prop < 0.8:
            log.info(
                'WARNING: keeping less than 80% of variance will have bad performance.')

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"{args.image} does not exist.")
    if not os.path.exists(args.sm_image):
        raise FileNotFoundError(f"{args.sm_image} does not exist.")
    if not os.path.exists(args.covar):
        raise FileNotFoundError(f"{args.covar} does not exist.")


def read_images_hdf5(dir):
    with h5py.File(dir, 'r') as file:
        images = file['images']
        ids = file['id'][:]
        coord = file['coord'][:]
    ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=['FID', 'IID'])
    return images, ids, coord


def get_batch_size(n_top, n_sub, n_voxels):
    """
    The rule to get batch size:
    if the maximum #PC < 10000, then use all subjects in one batch
    else if n_top > 5000, use n_top as the batch size
        else use n_sub.

    """
    max_n_pc = np.min((n_sub, n_voxels))
    if max_n_pc <= 10000:
        if n_sub <= 50000:
            return n_sub
        else:
            return n_sub // (n_sub // 50000 + 1)
    else:
        if n_top >= 5000:
            return n_top
        else:
            return n_sub


def run(args, log):
    # check input
    check_input(args, log)

    # read smoothed images
    log.info(f'Read smoothed images from {args.sm_image}')
    with h5py.File(args.sm_image, 'r') as file:
        sm_images = file['images']
        coord = file['coord'][:]
        # this id is only used to determine max n_components
        ids = file['id'][:]
        n_voxels, dim = coord.shape
        n_sub = len(ids)

        # incremental PCA
        log.info(f'Doing functional PCA ...')
        max_n_pc = np.min((n_sub, n_voxels))
        if args.all:
            n_top = max_n_pc
        elif args.n_ldrs:
            if args.n_ldrs > max_n_pc:
                n_top = max_n_pc
                log.info(
                    'WARNING: --n-ldrs is greater than the maximum #components.')
            else:
                n_top = args.n_ldrs
        else:
            if dim == 1:
                n_top = max_n_pc
            else:
                n_top = int(max_n_pc / (dim - 1))
        log.info(f"Computing the top {n_top} components.")

        batch_size = get_batch_size(n_top, n_sub, n_voxels)
        ipca = IncrementalPCA(n_components=n_top, batch_size=batch_size)
        max_avail_n_sub = n_sub // batch_size * batch_size
        log.info((f'The smoothed images are split into {n_sub // batch_size} batch(es), '
                  f'with batch size {batch_size}.'))

        for i in tqdm(range(0, max_avail_n_sub, batch_size), desc=f"{n_sub // batch_size} batch(es)"):
            ipca.partial_fit(sm_images[i: i+batch_size])
    values = ipca.singular_values_ ** 2
    eff_num = np.sum(values) ** 2 / np.sum(values ** 2)

    # generate LDR
    if args.prop:
        n_opt = determine_n_ldr(values, args.prop, log)
    else:
        n_opt = n_top

    log.info(f'Read raw images from {args.image} and construct {n_opt} LDRs.')
    with h5py.File(args.image, 'r') as file:
        images = file['images']
        # this id is used to take intersection with --covar and --keep
        ids = file['id'][:]
        ids = pd.MultiIndex.from_arrays(
            ids.astype(str).T, names=['FID', 'IID'])
        ldr = ipca.transform(images)[:, :n_opt]

    # read covariates
    log.info(f"Read covariates from {args.covar}")
    covar = Covar(args.covar, args.cat_covar_list)
    common_idxs = covar.data.index.intersection(ids)

    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects are common in --keep.')
        common_idxs = common_idxs.intersection(keep_idvs)  # slow

    # keep common subjects
    log.info(f'{len(common_idxs)} common subjects in these files.')
    covar.keep(common_idxs)
    covar.cat_covar_intercept()
    log.info(
        f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept).")
    ldr = ldr[ids.isin(common_idxs)]

    # var-cov matrix of projected LDRs
    proj_inner_ldr = projection_ldr(ldr, np.array(covar.data))
    log.info(f"Removed covariate effects from LDRs and computed inner product.\n")
    ldr_df = pd.DataFrame(ldr, index=common_idxs)

    # save the output
    ldr_df.to_csv(f"{args.out}_ldr_top{n_opt}.txt", sep='\t')
    np.save(f"{args.out}_proj_innerldr_top{n_opt}.npy", proj_inner_ldr)
    np.save(f"{args.out}_bases_top{n_opt}.npy", ipca.components_)
    np.save(f"{args.out}_eigenvalues_top{n_top}.npy", values)
    log.info((f"The effective number of independent voxels is {round(eff_num, 3)}, "
              f"which can be used in the Bonferroni p-value threshold (e.g., 0.05/{round(eff_num, 3)}) "
              "across all voxels."))
    log.info(f"Save the raw LDRs to {args.out}_ldr_top{n_opt}.txt")
    log.info(
        f"Save the projected inner product of LDRs to {args.out}_innerldr_top{n_opt}.npy")
    log.info(f"Save the top {n_opt} bases to {args.out}_bases_top{n_opt}.npy")
    log.info(
        f"Save the top {n_top} eigenvalues to {args.out}_eigenvalues_top{n_top}.npy")
