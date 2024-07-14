import h5py
import numpy as np
import heig.input.dataset as ds
from heig.wgs.utils import keep_ldrs


def check_input(args):
    # required arguments
    if args.bases is None:
        raise ValueError('--bases is required')
    if args.ldrs is None:
        raise ValueError('--ldrs is required')
    if args.covar is None:
        raise ValueError('--covar is required')
    # optional arguments
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0')


def run(args, log):
    # read ldrs, bases, and inner_ldr
    log.info(f"Read LDRs from {args.ldrs}")
    ldrs = ds.Dataset(args.ldrs)
    log.info(f'{ldrs.data.shape[1]} LDRs and {ldrs.data.shape[0]} subjects.')
    bases = np.load(args.bases)
    log.info(f'{bases.shape[1]} bases read from {args.bases}')
    
    # read covar
    log.info(f"Read covariates from {args.covar}")
    covar = ds.Covar(args.covar, args.cat_covar_list)

    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None

    common_idxs = ds.get_common_idxs(ldrs.data.index, covar.data.index, keep_idvs)
    log.info(f'{len(common_idxs)} subjects common in these files.')

    # keep common subjects
    covar.keep(common_idxs)
    covar.cat_covar_intercept()
    log.info(f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept).")
    ldrs.keep(common_idxs)
    ldrs.to_single_index()
    ids = ldrs.data.index
    covar.data = np.array(covar.data)
    ldrs.data = np.array(ldrs.data)

    # keep selected LDRs
    if args.n_ldrs is not None:
        bases, ldrs.data = keep_ldrs(args.n_ldrs, bases, ldrs.data)
        log.info(f'Keep the top {args.n_ldrs} LDRs.')
    
    # fit the null model
    log.info('Fitting the null model ...')
    inner_covar_inv = np.linalg.inv(np.dot(covar.data.T, covar.data)) # (X'X)^{-1} (p, p)
    covar_ldrs = np.dot(covar.data.T, ldrs.data) # X'\Xi (p, r)
    resid_ldr = ldrs.data - np.dot(covar.data, np.dot(inner_covar_inv, covar_ldrs)) # \Xi - X(X'X)^{-1}X'\Xi (n, r)
    # n, p = covar.data.shape
    # inner_ldr = np.dot(resid_ldr.T, resid_ldr) # (r, r)
    # var = np.sum(np.dot(bases, inner_ldr) * bases, axis=1) / (n - p)  # (N, )
    
    with h5py.File(f'{args.out}_null_model.h5', 'w') as file:
        file.create_dataset('covar', data=covar.data)
        file.create_dataset('resid_ldr', data=resid_ldr)
        # file.create_dataset('var', data=var)
        file.create_dataset('id', data=np.array(ids.tolist(), dtype='S10'))
    log.info(f'Save the null model to {args.out}_null_model.h5')