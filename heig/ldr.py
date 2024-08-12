import os
import h5py
import numpy as np
import pandas as pd
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
    inner_covar_inv = np.linalg.inv(inner_covar)
    ldr_covar = np.dot(ldr.T, covar)
    part2 = np.dot(np.dot(ldr_covar, inner_covar_inv), ldr_covar.T)
    
    return inner_ldr - part2


def check_input(args):
    # required arguments
    if args.image is None:
        raise ValueError('--image is required')
    if args.covar is None:
        raise ValueError('--covar is required')
    if args.bases is None:
        raise ValueError('--bases is required')
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0')
    
    # required arguments must exist
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"{args.image} does not exist")
    if not os.path.exists(args.covar):
        raise FileNotFoundError(f"{args.covar} does not exist")
    if not os.path.exists(args.bases):
        raise FileNotFoundError(f"{args.bases} does not exist")


def run(args, log):
    check_input(args)

    # read bases and extract top n_ldrs
    bases = np.load(args.bases)
    n_voxels, n_bases = bases.shape
    log.info(f'{n_bases} bases of {n_voxels} voxels (vertices) read from {args.bases}')

    if args.n_ldrs is not None:
        if args.n_ldrs <= n_bases:
            n_ldrs = args.n_ldrs
            bases = bases[:, :n_ldrs]
        else:
            raise ValueError('the number of bases is less than --n-ldrs')
    else:
        n_ldrs = n_bases

    # read images
    log.info(f'Read raw images from {args.image}')
    with h5py.File(args.image, 'r') as file:
        images = file['images']
        ids = file['id'][:]
        ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=['FID', 'IID'])
        if n_voxels != images.shape[1]:
            raise ValueError('the images and bases have different resolution')

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
        common_idxs = ds.get_common_idxs(ids, covar.data.index, keep_idvs)
        log.info(f'{len(common_idxs)} common subjects in these files.')

        # contruct ldrs
        ids_ = ids.isin(common_idxs)
        id_idxs = np.arange(len(ids))[ids_]
        ldrs = np.zeros((len(id_idxs), n_ldrs))
        for i, id_idx in enumerate(id_idxs):
            ldrs[i] = np.dot(images[id_idx], bases)
        log.info(f'{n_ldrs} LDRs constructed.')

    # process covar
    covar.keep(common_idxs)
    covar.cat_covar_intercept()
    log.info(f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept).")

    # var-cov matrix of projected LDRs
    proj_inner_ldr = projection_ldr(ldrs, np.array(covar.data))
    log.info(f"Removed covariate effects from LDRs and computed inner product.\n")

    # save the output
    ldr_df = pd.DataFrame(ldrs, index=ids[ids_])
    ldr_df.to_csv(f"{args.out}_ldr_top{n_ldrs}.txt", sep='\t')
    np.save(f"{args.out}_proj_innerldr_top{n_ldrs}.npy", proj_inner_ldr)

    log.info(f"Save the raw LDRs to {args.out}_ldr_top{n_ldrs}.txt")
    log.info(f"Save the projected inner product of LDRs to {args.out}_proj_innerldr_top{n_ldrs}.npy")