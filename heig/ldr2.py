import sys, os
import pickle
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from numpy.linalg import inv
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csc_matrix, dok_matrix, csr_matrix, hstack
from . import utils



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
                mse[cii] = np.mean(np.sum((self.data - y_sm) ** 2,
                           axis=1) / (1 - np.sum(csc_matrix.diagonal(sparse_sm_weight)) / self.N) ** 2)
                log.info(f"The MSE for bandwidth {np.round(bw, 3)} is {round(mse[cii], 3)}.")
            else:
                mse[cii] = np.Inf
        
        which_min = np.argmin(mse)
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
        self.t_mat0, self.t_mat = self._initial_weight()
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
        # bw = bw.reshape(self.d, 1, 1)
        sparse_sm_weight = dok_matrix((self.N, self.N), dtype=np.float32)
        for lii in range(self.N):
            # k_mat = np.zeros((self.N, self.d))
            t_mat0 = self.coord - self.coord[lii] # N * d
            t_mat = np.hstack((np.ones(self.N).reshape(-1, 1), t_mat0))
            dis = t_mat0 / bw
            close_points = (dis < 4) & (dis > -4)
            # k_mat[close_points] = self._gau_kernel(dis[close_points])
            k_mat = csr_matrix((self._gau_kernel(dis[close_points]), np.where(close_points)), (self.N, self.d))
            k_mat = csc_matrix(np.prod(k_mat / bw, axis=1)) # can be faster
            # temp = np.repeat(k_mat, self.d + 1).reshape(self.N, self.d + 1)  # N * (d+1)
            # temp_nonzero_idxs = np.where(temp > 0)
            # k_mat_sparse = csc_matrix((temp[temp_nonzero_idxs], temp_nonzero_idxs), shape=(self.N, self.d + 1))  # N * (d+1)
            k_mat_sparse = hstack([k_mat] * (self.d + 1))
            kx = k_mat_sparse.multiply(t_mat).T # (d+1) * N
            sm_weight = inv(kx @ t_mat + np.eye(self.d + 1) * 0.000001)[0, :] @ kx # N * 1
            large_weight_idxs = np.where(np.abs(sm_weight) > 1 / self.N)
            sparse_sm_weight[lii, large_weight_idxs] = sm_weight[large_weight_idxs]
        nonzero_weights = np.sum(sparse_sm_weight != 0, axis=0)
        if np.mean(nonzero_weights) > self.N // 10:
            self.logger.info(f"On average, the non-zero weights are greater than #voxels // 10 ({self.N // 10}). Skip.")
            return False, None, None

        # sm_weight = np.ones((self.N, self.N))
        # k_mat = np.zeros((self.d, self.N, self.N))

        # bw = bw.reshape(self.d, 1, 1)
        # dis = self.t_mat0 / bw # d * N * N 
        # close_points = (dis < 4) & (dis > -4)
        # k_mat[close_points] = self._gau_kernel(dis[close_points])
        # k_mat = np.prod(k_mat / bw, axis=0) # N * N

        # for lii in range(self.N):
        #     temp = np.repeat(k_mat[:, lii], self.d + 1).reshape(self.N, self.d + 1) # N * (d+1)
        #     temp_nonzero_idxs = np.where(temp > 0)
        #     k_mat_sparse = csc_matrix((temp[temp_nonzero_idxs], temp_nonzero_idxs), shape=(self.N, self.d + 1)) # N * (d+1)
        #     kx = k_mat_sparse.multiply(self.t_mat[lii, :, :]).T # (d+1) * N
        #     sm_weight[lii, ] = inv(kx @ self.t_mat[lii, :, :] + np.eye(self.d + 1) * 0.000001)[0, :] @ kx # N * 1

        # sparse_thresh = np.abs(np.max(sm_weight) / self.N)  # a threhold to make it sparse
        # large_weight_idxs = np.where(np.abs(sm_weight) > sparse_thresh)
        # sparse_sm_weight = csc_matrix((sm_weight[large_weight_idxs],
        #                            large_weight_idxs), shape=(self.N, self.N))
        # nonzero_weights = np.sum(sparse_sm_weight != 0, axis=0)
        # if np.mean(nonzero_weights) > self.N // 10:
        #     self.logger.info(f"On average, the non-zero weights are greater than #voxels // 10 ({self.N // 10}). Skip.")
        #     return False, None, None
        
        sm_data = self.data @ sparse_sm_weight.T
        return True, sm_data, sparse_sm_weight
    

    def _initial_weight(self):
        """
        Generating initial weights 
        
        """
        t_mat0 = np.zeros((self.d + 1, self.N, self.N))
        t_mat0[0, :, :] = np.ones((self.N, self.N))

        for dii in range(self.d):
        # di * J_{Nv}.T - J_{Nv} * di.T -> Nv*Nv
            temp_dii = np.repeat(self.coord[:, dii], self.N).reshape(self.N, self.N)
            t_mat0[dii + 1, :, :] = temp_dii - temp_dii.T
            
        t_mat = np.transpose(t_mat0, [2, 1, 0])
        t_mat0 = t_mat0[1:]

        return t_mat0, t_mat
    

    
class LocalConstant(KernelSmooth):
    def smoother(self, bw):
        """
        Local linear smoother. 
        idea of the algorithm:
        1. compute pairwise distance once for the fisrt point
        2. fill in only the upper triangle of the sm matrix by sliding right one index each time 
        from the previous point
        3. map the upper triangle to lower since it is a symmetric matrix
        4. make it sparse to speed up matrix production

        Parameters:
        ------------
        bw (d * 1): bandwidth for d dimension
        
        Returns:
        ---------
        If this bandwidth is valid to make the weight matrix sparse;
        The smooothed data, and the sparse weights.

        """
        dis = [self.coord[0] - self.coord[i] for i in range(len(self.coord))]
        sm_weight = np.zeros((self.N, self.N))

        init_weight = np.ones((1, self.N))
        for dii in range(self.d):
            init_weight *= self._gau_kernel(dis / bw[dii], bw[dii])
        sm_weight[0, :] = np.squeeze(init_weight)

        for i in range(1, self.N):
            sm_weight[i, i:] = sm_weight[0, i:]

        sm_weight = (sm_weight + sm_weight.T - np.diag(np.diag(sm_weight))) / self.N

        sparse_thresh = np.abs(np.max(sm_weight) / self.N)
        sparse_sm_weight = csc_matrix((sm_weight[np.abs(sm_weight) > sparse_thresh],
                                    np.where(np.abs(sm_weight) > sparse_thresh)), shape=(self.N, self.N))
        # sm_data = np.dot(self.data, sparse_sm_weight.T)
        nonzero_weights = np.sum(sparse_sm_weight != 0, axis=0)
        if np.mean(nonzero_weights) > self.N // 20:
            self.logger.info(f"On average, the non-zero weights are greater than {self.N // 20}. Skip.")
            return None, None

        sm_data = self.data @ sparse_sm_weight.T
        
        return sm_data, sparse_sm_weight



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
                                    na_values=[-9, 'NONE', '.']) # -9.0 is not counted TODO: test it
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
            self.logger.info(f"There are {len(catlist)} categorical variables provided by --cat-covar-list.")
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


def read_images(img_file, suffix,):
    """
    Reading a single image
    
    Parameters:
    ------------
    dir: folder containing all images with the name convension id.suffix
    suffix: suffix of images
    
    Returns:
    ---------
    id: subject id
    coord: coordinates of nonzero values 
    data: nonzero values

    """
    # img_files = [img_file for img_file in os.listdir(dir) if img_file.endswith(suffix)]
    # log.info(f"{len(img_files)} images with suffix {suffix} in folder {dir}.")
    # keep_idv = set(keep_idv)
    
    id = img_file.replace(suffix, '')
    img = nib.load(os.path.join(dir, img_file))
    data = img.get_fdata()
    coord = np.stack(np.nonzero(data))
    data = data[data != 0]
    
    return id, coord, data
    


def get_common_idxs(dataset1, dataset2):
    """
    Getting common indices of two Datasets

    Parameters:
    ------------
    dataset1: an instance of class Dataset
    dataset2: an instance of class Dataset

    Returns:
    ---------
    common_idxs: common indices
    
    """
    if not isinstance(dataset1, Dataset) or not isinstance(dataset2, Dataset):
        raise ValueError('The input should be an instance of Dataset.')
    common_idxs = dataset1.data.index.intersection(dataset2.data.index)
    # sorted(common_idxs)
    return common_idxs



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


def select_n_ldr(data, bases, log):
    N, n_bases = bases.shape
    step = int(n_bases / 5)
    n_cand_list = list(range(step, n_bases, step))
    
    log.info(f'Selecting the optimal number of components from {n_cand_list} ...')
    mse = []
    for n_cand in n_cand_list:
        bases_n = bases[:, :n_cand]
        y_hat = np.dot(np.dot(data, bases_n), bases_n.T)
        mse.append(np.mean(np.sum((data - y_hat) ** 2,
                       axis=1) / (1 - np.sum(bases_n ** 2) / N) ** 2))
        log.info(f'The MSE of {n_cand} is {round(mse[-1], 3)}')
        
    n_opt = n_cand_list[np.argmin(mse)]
    log.info(f'The optimal number of components is {n_opt}.')
    return n_opt


def determine_n_ldr(values, prop, log):
    eff_num = np.sum(values) ** 2 / np.sum(values ** 2)
    prop_var = np.cumsum(values) / np.sum(values)
    idxs = (prop_var <= prop) & (values != 0)
    n_idxs = np.sum(idxs) + 1
    n_opt = max(n_idxs, int(eff_num) + 1)
    var_prop = np.sum(values[:n_opt]) / np.sum(values)
    log.info(f'Approximately {round(var_prop * 100, 1)}% variance is captured by the top {n_opt} LDRs.\n')
    # if n_opt == n_idxs:
    #     log.info((f'The analysis indicates that the decay rate of eigenvalues is low, '
    #               'then the downstream heritability and genetic correlation analysis may be noisy.'))
    return n_opt


def check_input(args, log):
    if not args.image:
        raise ValueError('--image is required.')
    if not args.coord:
        raise ValueError('--coord is required.')
    if not args.covar:
        raise ValueError('--covar is required.')
    if not args.out:
        raise ValueError('--out is required.')
    if not args.prop:
        args.prop = 0.8
        log.info("By default, perserving 80% of variance.")

    if not os.path.exists(args.image):
        raise ValueError(f"{args.image} does not exist.") 
    if not os.path.exists(args.coord):
        raise ValueError(f"{args.coord} does not exist.") 
    if not os.path.exists(args.covar):
        raise ValueError(f"{args.covar} does not exist.")
    if args.keep and not os.path.exists(args.keep):
        raise ValueError(f"{args.covar} does not exist.")
    if args.prop and (args.prop <= 0 or args.prop > 1):
        raise ValueError('--prop should be between 0 and 1.')
    if args.prop and args.prop < 0.8:
        log.info('WARNING: keeping less than 80% of variance will have bad performance.')
    if args.bw_opt and args.bw_opt <= 0:
        raise ValueError('--bw-opt should be positive.')
        

def run(args, log):
    # check input
    check_input(args, log)

    # read images
    log.info(f"Reading imaging data from {args.image}")
    image = Dataset(args.image)
    log.info(f"{image.data.shape[0]} subjects and {image.data.shape[1]} voxels are included in the data.")

    # read coordinate
    log.info(f"Reading coordinate data from {args.coord}")
    coord = np.loadtxt(args.coord)
    if np.isnan(coord).any():
        raise ValueError('Missing data is not allowed in the coordinate.')
    if image.data.shape[1] != coord.shape[0]:
        raise ValueError('Data and coordinates have inconsistent voxels.')
    
    # read covariates
    log.info(f"Reading covariates from {args.covar}")
    covar = Covar(args.covar, args.cat_covar_list)
    log.info(f"There are {covar.data.shape[1]} fixed effects in the covariates (including the intercept).")

    # extract common subjects
    common_idxs = get_common_idxs(image, covar)
    if args.keep:
        keep_ids = pd.read_csv(args.keep, delim_whitespace=True, header=None, usecols=[0, 1])
        keep_ids = list(zip(keep_ids[0], keep_ids[1]))
        common_idxs = common_idxs.intersection(tuple(keep_ids)) # slow
    image.keep(common_idxs)
    covar.keep(common_idxs)
    log.info(f"{len(common_idxs)} subjects are common in images and covariates.\n")

    # kernel smoothing
    log.info('Doing kernel smoothing using the local linear method ...')
    data = np.array(image.data)
    sm_data = do_kernel_smoothing(data, coord, args.bw_opt, log)
        
    # eigen decomposion 
    n_points, dim = coord.shape
    if args.all:
        n_top = n_points
        log.info(f"Computing all {n_top} components, which may take longer time.")
    else:
        if dim == 1:
            n_top = int(n_points / 4)
        else:
            n_top = int(n_points ** ((dim - 1) / dim)) 
        log.info(f"Computing only the first {n_top} components.")
    log.info(f'Adaptively determining the number of low-dimension representations (LDRs) ...')
    values, bases = functional_bases(sm_data, n_top)
    # n_opt = select_n_ldr(sm_data, bases)
    n_opt = determine_n_ldr(values, args.prop, log) # keep at least 80% variance
    bases = bases[:, :n_opt]
    eff_num = np.sum(values) ** 2 / np.sum(values ** 2)

    # generate LDR
    ldr = np.dot(data, bases)
    proj_inner_ldr = projection_ldr(ldr, np.array(covar.data))
    ldr_df = pd.DataFrame(ldr, index=image.data.index)

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
    
