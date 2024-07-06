import h5py
import numpy as np
import heig.input.dataset as ds


def check_input(args):
    # required arguments
    if args.bases is None:
        raise ValueError('--bases is required')
    if args.inner_ldr is None:
        raise ValueError('--inner-ldr is required')
    if args.ldrs is None:
        raise ValueError('--ldrs is required')
    if args.covar is None:
        raise ValueError('--covar is required')
    # optional arguments
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0')


def keep_ldrs(n_ldrs, bases, inner_ldr, ldrs):
    if bases.shape[1] < n_ldrs:
        raise ValueError('the number of bases is less than --n-ldrs')
    if inner_ldr.shape[0] < n_ldrs:
        raise ValueError('the dimension of inner product of LDR is less than --n-ldrs')
    if ldrs.shape[1] < n_ldrs:
        raise ValueError('the number of LDRs is less than --n-ldrs')
    bases = bases[:, :n_ldrs]
    inner_ldr = inner_ldr[:n_ldrs, :n_ldrs]
    ldrs = ldrs[:, :n_ldrs]

    return bases, inner_ldr, ldrs


def run(args, log):
    # read ldrs, bases, and inner_ldr
    ldrs = ds.Dataset(args.ldrs)
    log.info(f'{ldrs.shape[1]} LDRs of {ldrs.shape[0]} subjects read from {args.ldrs}')
    bases = np.load(args.bases)
    log.info(f'{bases.shape[1]} bases read from {args.bases}')
    inner_ldr = np.load(args.inner_ldr)
    log.info(f'Read inner product of LDRs from {args.inner_ldr}')

    # keep selected LDRs
    if args.n_ldrs is not None:
        bases, inner_ldr, ldrs = keep_ldrs(args.n_ldrs, bases, inner_ldr, args.ldrs)
        log.info(f'Keep the top {args.n_ldrs} LDRs.')
    
    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None
    
    # read covar and keep common subjects
    covar = ds.Covar(args.covar, args.cat_covar_list)
    common_idxs = ds.get_common_idxs(ldrs.data.index, covar.data.index, keep_idvs)
    covar.keep(common_idxs)
    covar.cat_covar_intercept()
    ldrs.keep(common_idxs)
    ldrs.to_single_index()
    ids = ldrs.data.index
    log.info(f'{len(ldrs)} subjects common in these files.')
    
    # fit the null model
    log.info('Fitting the null model ...')
    covar.data = np.array(covar.data)
    ldrs.data = np.array(ldrs.data)
    n, p = covar.data.shape
    inner_covar_inv = np.linalg.inv(np.dot(covar.data.T, covar.data)) # (X'X)^{-1} (p, p)
    covar_ldrs = np.dot(covar.data.T, ldrs.data) # X'\Xi (p, r)
    resid_ldr = ldrs.data - np.dot(covar.data, np.dot(inner_covar_inv, covar_ldrs)) # \Xi - X(X'X)^{-1}X'\Xi (n, r)
    var = np.sum(np.dot(bases, inner_ldr) * bases, axis=1) / (n - p)  # (N, )
    
    log.info(f'Save the null model to {args.out}_null_model.h5')
    with h5py.File(f'{args.out}_null_model.h5', 'w') as file:
        file.create_dataset('covar', data=covar.data)
        file.create_dataset('resid_ldr', data=resid_ldr)
        file.create_dataset('var', data=var)
        file.create_dataset('id', data=ids)