import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import heig.input.dataset as ds


def projection_ldr(ldr, covar):
    """
    Computing S'(I - M)S = S'S - S'X(X'X)^{-1}X'S,
    where I is the identity matrix, 
    M = X(X'X)^{-1}X' is the project matrix for X,
    S is the LDR matrix.

    Parameters:
    ------------
    ldr (N, r): low-dimension representaion of imaging data
    covar (n, p): covariates, including the intercept

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
    log.info(
        f'Approximately {round(var_prop * 100, 1)}% variance is captured by the top {n_opt} components.\n')
    return n_opt


def check_input(args, log):
    if args.image is None:
        raise ValueError('--image is required')
    if args.sm_image is None:
        raise ValueError('--sm-image is required')
    if args.covar is None:
        raise ValueError('--covar is required')
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
            log.info(
                'WARNING: keeping less than 80% of variance will have bad performance.')

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"{args.image} does not exist")
    if not os.path.exists(args.sm_image):
        raise FileNotFoundError(f"{args.sm_image} does not exist")
    if not os.path.exists(args.covar):
        raise FileNotFoundError(f"{args.covar} does not exist")


def get_batch_size(n_top, n_sub, n_voxels):
    """
    Adaptively determine batch size

    Parameters:
    ------------
    n_top: the number of top components to compute in PCA
    n_sub: the sample size
    n_voxel: resolution of images

    Returns:
    ---------
    batch size for IncrementalPCA

    """
    max_n_pc = np.min((n_sub, n_voxels))
    if max_n_pc <= 15000:
        if n_sub <= 50000:
            return n_sub
        else:
            return n_sub // (n_sub // 50000 + 1)
    else:
        if n_top > 15000 or n_sub > 50000:
            i = 2
            while n_sub // i > 50000:
                i += 1
            return n_sub // i
        else:
            return n_sub


def get_n_top(n_ldrs, max_n_pc, n_sub, dim, all, log):
    """
    Determine the number of top components to compute in PCA.

    Parameters:
    ------------
    n_ldrs: a specified number of components
    max_n_pc: the maximum number of components
    n_sub: sample size
    dim: the dimension of images
    all: a boolean variable for computing all components
    log: a logger

    Returns:
    ---------
    n_top: the number of top components to compute in PCA

    """
    if all:
        n_top = max_n_pc
    elif n_ldrs is not None:
        if n_ldrs > max_n_pc:
            n_top = max_n_pc
            log.info(
                'WARNING: --n-ldrs is greater than the maximum #components.')
        else:
            n_top = n_ldrs
    else:
        if dim == 1:
            n_top = max_n_pc
        else:
            n_top = int(max_n_pc / (dim - 1))

    n_top = np.min((n_top, n_sub))

    return n_top


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

        # setup parameters
        log.info(f'Doing functional PCA ...')
        max_n_pc = np.min((n_sub, n_voxels))
        n_top = get_n_top(args.n_ldrs, max_n_pc, n_sub, dim, args.all, log)
        log.info(f"Computing the top {n_top} components.")
        batch_size = get_batch_size(n_top, n_sub, n_voxels)

        # incremental PCA
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
        coord = file['coord'][:]
        if coord.shape != (n_voxels, dim):
            raise ValueError(
                'the smoothed images and raw images have different resolution')
        # this id is used to take intersection with --covar and --keep
        ids = file['id'][:]
        ids = pd.MultiIndex.from_arrays(
            ids.astype(str).T, names=['FID', 'IID'])
        ldr = ipca.transform(images)[:, :n_opt]

    # read covariates
    log.info(f"Read covariates from {args.covar}")
    covar = ds.Covar(args.covar, args.cat_covar_list)

    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None

    # keep common subjects
    common_idxs = ds.get_common_idxs(covar.data.index, ids, keep_idvs)
    log.info(f'{len(common_idxs)} common subjects in these files.')
    covar.keep(common_idxs)
    covar.cat_covar_intercept()
    log.info(
        f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept).")
    ldr = ldr[ids.isin(common_idxs)]

    # var-cov matrix of projected LDRs
    proj_inner_ldr = projection_ldr(ldr, np.array(covar.data))
    log.info(f"Removed covariate effects from LDRs and computed inner product.\n")

    # save the output
    ldr_df = pd.DataFrame(ldr, index=common_idxs)
    ldr_df.to_csv(f"{args.out}_ldr_top{n_opt}.txt", sep='\t')
    np.save(f"{args.out}_proj_innerldr_top{n_opt}.npy", proj_inner_ldr)
    np.save(f"{args.out}_bases_top{n_opt}.npy", ipca.components_.T)
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
