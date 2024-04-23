import sys, os
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from . import utils


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
                                    na_values=[-9, 'NONE', '.'], dtype={'FID': str, 'IID': str}) # -9.0 is not counted TODO: test it
        if self.data[['FID', 'IID']].duplicated().any():
            first_dup = self.data.loc[self.data[['FID', 'IID']].duplicated(), ['FID', 'IID']]
            raise ValueError(f'Subject {list(first_dup)} is duplicated.')
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
            raise ValueError('Only one column detected, check your input file and delimiter.')
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
        self.logger.info(f"{sum(bad_idxs)} row(s) with missing or infinite values were removed.")
    

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

        if cat_covar_list:
            catlist = cat_covar_list.split(',')
            self._check_validcatlist(catlist)
            self.logger.info(f"{len(catlist)} categorical variables provided by --cat-covar-list.")
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
        if self._check_singularity():
            raise ValueError('The covarite matrix is singular.')
        


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
    inner_covar_inv = np.linalg.inv(inner_covar) # nonsingularity has been checked
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
    log.info(f'Approximately {round(var_prop * 100, 1)}% variance is captured by the top {n_opt} LDRs.\n')
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
    if args.prop is not None:
        if args.prop <= 0 or args.prop > 1:
            raise ValueError('--prop should be between 0 and 1.')
        elif args.prop < 0.8:   
            log.info('WARNING: keeping less than 80% of variance will have bad performance.')

    if not os.path.exists(args.image):
        raise ValueError(f"{args.image} does not exist.") 
    if not os.path.exists(args.sm_image):
        raise ValueError(f"{args.sm_image} does not exist.") 
    if not os.path.exists(args.covar):
        raise ValueError(f"{args.covar} does not exist.")
    if args.keep is not None and not os.path.exists(args.keep):
        raise ValueError(f"{args.keep} does not exist.")
    
    

def read_images_hdf5(dir):
    with h5py.File(dir, 'r') as file:
        images = file['images']
        ids = file['id'][:]
        coord = file['coord'][:]
    ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=['FID', 'IID'])
    return images, ids, coord



def run(args, log):
    # check input
    check_input(args, log)
    
    # read covariates
    log.info(f"Reading covariates from {args.covar}")
    covar = Covar(args.covar, args.cat_covar_list)
    log.info(f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept).")
    common_idxs = covar.data.index

    # extract common subjects
    if args.keep:
        keep_ids = pd.read_csv(args.keep, delim_whitespace=True, header=None, usecols=[0, 1],
                               dtype={0: str, 1: str})
        keep_ids = tuple(zip(keep_ids[0], keep_ids[1]))
        log.info(f'{len(keep_ids)} subjects are in {args.keep}')
        common_idxs = covar.data.index.intersection(keep_ids) # slow
    # log.info(f"{len(common_idxs)} subjects are common in images and covariates.\n")

    # read images
    # coord = pd.read_csv(args.coord, delim_whitespace=True, header=None)
    # ids = pd.read_csv(args.id, delim_whitespace=True, header=None)
    # n_voxels, dim = coord.shape
    # n_sub = len(ids)
    # mm = np.memmap(args.sm_image, dtype='float32', mode='r', shape=(n_sub, n_voxels))
    
    # read smoothed images
    # sm_images, ids, coord = read_images_hdf5(args.sm_image)
    with h5py.File(args.sm_image, 'r') as file:
        sm_images = file['images']
        ids = file['id'][:]
        coord = file['coord'][:]
        ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=['FID', 'IID'])
        n_voxels, dim = coord.shape
        n_sub = len(ids)
        common_idxs = common_idxs.intersection(ids)
        covar.keep(common_idxs)

        # incremental PCA 
        if args.all:
            n_top = n_voxels
        elif args.n_ldrs:
            n_top = args.n_ldrs
        elif args.prop:
            if dim == 1:
                n_top = np.min((n_sub, n_voxels))
            else:
                n_top = int(np.min((n_sub, n_voxels)) / (dim - 1))
        else:
            n_top = int(np.min((n_sub, n_voxels)) / 10)
        log.info(f"Computing the top {n_top} components.")
        
        batch_size = np.max((n_top, 500))
        ipca = IncrementalPCA(n_components=n_top, batch_size=batch_size)
        max_avail_n_sub = n_sub // batch_size * batch_size
        for i in range(0, max_avail_n_sub, batch_size):
            ipca.partial_fit(sm_images[i: i+batch_size])
    values = ipca.singular_values_ ** 2
    eff_num = np.sum(values) ** 2 / np.sum(values ** 2)

    # generate LDR
    if args.n_ldrs:
        n_opt = args.n_ldrs
    elif args.prop:
        n_opt = determine_n_ldr(values, args.prop, log) # keep at least 80% variance
    else:
        n_opt = np.max((n_sub, n_voxels)) # an arbitrary large number
    # images, ids, coord = read_images_hdf5(args.image)
    with h5py.File(args.image, 'r') as file:
        images = file['images']
        ldr = ipca.transform(images)[ids.isin(common_idxs), :n_opt]

    proj_inner_ldr = projection_ldr(ldr, np.array(covar.data))
    ldr_df = pd.DataFrame(ldr, index=covar.data.index)

    # save the output
    ldr_df.to_csv(f"{args.out}_ldr_top{n_opt}.txt", sep='\t')
    np.save(f"{args.out}_proj_innerldr_top{n_opt}.npy", proj_inner_ldr)
    np.save(f"{args.out}_bases_top{n_opt}.npy", ipca.components_)
    np.save(f"{args.out}_eigenvalues_top{n_top}.npy", values)
    log.info((f"The effective number of independent voxels is {round(eff_num, 3)}, "
              "which can be used in the Bonferroni p-value threshold across all voxels."))
    log.info(f"Save the LDRs to {args.out}_ldr_top{n_opt}.txt")
    log.info(f"Save the inner product of LDRs to {args.out}_innerldr_top{n_opt}.npy")
    log.info(f"Save the top {n_opt} bases to {args.out}_bases_top{n_opt}.npy")
    log.info(f"Save the top {n_top} eigenvalues to {args.out}_eigenvalues_top{n_top}.npy")