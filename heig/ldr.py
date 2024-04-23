import sys, os
import pickle
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from numpy.linalg import inv
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, hstack
from . import utils


"""
TODO: 
add an option for n_ldrs
add a function to read FreeSurfer data
add a step that saves the large imaging matrix in disk and then do randomized svd

"""

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
                mse[cii] = np.mean(np.sum((self.data - y_sm) ** 2, axis=1)) / (1 - np.sum(sparse_sm_weight.diagonal()) / self.N + 10**-10) ** 2
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
        sparse_sm_weight = dok_matrix((self.N, self.N), dtype=np.float64)
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



def load_images(img_files, log):
    for i, img_file in enumerate(img_files):
        img = nib.load(img_file)
        data = img.get_fdata()
        if i == 0:
            idxs = data != 0 
            coord = np.stack(np.nonzero(data)).T
            n_voxels = np.sum(idxs)
            image = np.zeros((len(img_files), n_voxels), dtype=np.float64)
        image[i] = data[idxs]
        if i % 1000 == 1 and i > 0:
            log.info(f'Read {i+1} images.')

    return image, coord



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


def functional_bases(data, top):
    """
    Generating the top eigenvalues and eigenfunctions using randomized SVD

    Parameters:
    ------------
    data (n * N): smoothed imaging data
    
    Returns:
    ---------
    Top eigenvalues and eigenfunctions
    
    """
    data = data - np.mean(data, axis=0)
    _, s, vt = randomized_svd(data, top)
    
    return s ** 2, vt.T


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
    if (not (args.image_dir is not None and args.image_suffix is not None) and 
        not (args.image is not None and args.coord is not None)):
        raise ValueError('Either --image-dir + --image-suffix or --image + --coord is required.')
    elif (args.image_dir is not None and args.image_suffix is not None and
          args.image is not None and args.coord is not None):
        log.info('WARNING: --image-dir and --image-suffix are ignored.')
    if args.covar is None:
        raise ValueError('--covar is required.')
    if args.out is None:
        raise ValueError('--out is required.')
    if args.all:
        log.info('WARNING: computing all principal components might be very time and memory consuming.')
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0.')
    if args.prop is None:
        args.prop = 0.8
        log.info("By default, perserving 80% of variance.")
    elif args.prop <= 0 or args.prop > 1:
        raise ValueError('--prop should be between 0 and 1.')
    elif args.prop < 0.8:   
        log.info('WARNING: keeping less than 80% of variance will have bad performance.')
    if args.bw_opt and args.bw_opt <= 0:
        raise ValueError('--bw-opt should be positive.')

    if args.image_dir is not None and not os.path.exists(args.image_dir):
        raise ValueError(f"{args.image_dir} does not exist.") 
    if args.image is not None and not os.path.exists(args.image):
        raise ValueError(f"{args.image} does not exist.") 
    if args.coord is not None and not os.path.exists(args.coord):
        raise ValueError(f"{args.coord} does not exist.") 
    if not os.path.exists(args.covar):
        raise ValueError(f"{args.covar} does not exist.")
    if args.keep is not None and not os.path.exists(args.keep):
        raise ValueError(f"{args.covar} does not exist.")
    
        
    
def get_image_list(common_id, image_dir, suffix):
    ids = []
    img_files = []
    
    for img_file in os.listdir(image_dir):
        image_id = img_file.replace(suffix, '')
        if img_file.endswith(suffix) and image_id in common_id:
            ids.append(image_id)
            img_files.append(os.path.join(image_dir, img_file))
    img_files.sort()
    ids = pd.MultiIndex.from_arrays([ids, ids], names=['FID', 'IID'])
    
    return ids, img_files 
    

def run(args, log):
    # check input
    check_input(args, log)
    
    # read covariates
    log.info(f"Reading covariates from {args.covar}")
    covar = Covar(args.covar, args.cat_covar_list)
    log.info(f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept).")

    # extract common subjects
    if args.keep:
        keep_ids = pd.read_csv(args.keep, delim_whitespace=True, header=None, usecols=[0, 1],
                               dtype={0: str, 1: str})
        keep_ids = tuple(zip(keep_ids[0], keep_ids[1]))
        log.info(f'{len(keep_ids)} subjects are in {args.keep}')
        common_idxs = covar.data.index.intersection(keep_ids) # slow
    else:
        common_idxs = covar.data.index
    # log.info(f"{len(common_idxs)} subjects are common in images and covariates.\n")

    # read images
    if args.image is not None and args.coord is not None:
        log.info(f"Reading images from {args.image} ...")
        image = Dataset(args.image)
        log.info(f"Reading coordinate data from {args.coord}")
        coord = np.loadtxt(args.coord)
        if np.isnan(coord).any():
            raise ValueError('Missing data is not allowed in the coordinate.')
        if image.data.shape[1] != coord.shape[0]:
            raise ValueError('Data and coordinates have inconsistent voxels.')
        common_idxs = common_idxs.intersection(image.data.index)
        covar.keep(common_idxs)
        covar._check_singularity()
        image.keep(common_idxs)
        image = np.array(image.data, dtype=np.float64)
    else:
        common_idxs = common_idxs.get_level_values(0)
        common_idxs, img_files = get_image_list(common_idxs, args.image_dir, args.image_suffix)
        covar.keep(common_idxs)
        covar._check_singularity()
        log.info(f"{len(common_idxs)} subjects are common.\n")
        log.info(f'Reading images from {args.image_dir} ...')
        image, coord = load_images(img_files, log)
    log.info(f"{image.shape[0]} subjects and {image.shape[1]} voxels are included in the imaging data.")
    
    # kernel smoothing
    log.info('Doing kernel smoothing using the local linear method ...')
    sm_data = do_kernel_smoothing(image, coord, args.bw_opt, log)
    np.save(f"{args.out}_sm_images.npy", sm_data)
    log.info(f'Save the smoothed images to {args.out}_sm_images.npy')
        
    # SVD 
    n_points, dim = coord.shape
    if args.all:
        n_top = n_points
    elif args.n_ldrs:
        n_top = args.n_ldrs
    else:
        if dim == 1:
            n_top = np.min(image.shape)
        else:
            n_top = int(np.min(image.shape) / (dim - 1))
    log.info(f"Computing the top {n_top} components.")
    values, bases = functional_bases(sm_data, n_top)
    # n_opt = select_n_ldr(sm_data, bases)
    if args.n_ldrs:
        n_opt = args.n_ldrs
    else:
        n_opt = determine_n_ldr(values, args.prop, log) # keep at least 80% variance
    bases = bases[:, :n_opt]
    eff_num = np.sum(values) ** 2 / np.sum(values ** 2)

    # generate LDR
    ldr = np.dot(image, bases)
    proj_inner_ldr = projection_ldr(ldr, np.array(covar.data))
    ldr_df = pd.DataFrame(ldr, index=covar.data.index)

    # save the output
    ldr_df.to_csv(f"{args.out}_ldr_top{n_opt}.txt", sep='\t')
    np.save(f"{args.out}_proj_innerldr_top{n_opt}.npy", proj_inner_ldr)
    np.save(f"{args.out}_bases_top{n_opt}.npy", bases)
    np.save(f"{args.out}_eigenvalues_top{n_top}.npy", values)
    log.info((f"The effective number of independent voxels is {round(eff_num, 3)}, "
              "which can be used in the Bonferroni p-value threshold across all voxels."))
    log.info(f"Save the LDRs to {args.out}_ldr_top{n_opt}.txt")
    log.info(f"Save the inner product of LDRs to {args.out}_innerldr_top{n_opt}.npy")
    log.info(f"Save the top {n_opt} bases to {args.out}_bases_top{n_opt}.npy")
    log.info(f"Save the top {n_top} eigenvalues to {args.out}_eigenvalues_top{n_top}.npy")
    
