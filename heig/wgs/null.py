import h5py
import numpy as np
import heig.input.dataset as ds


def check_input(args, log):
    pass


def run(args, log):
    # read ldrs, bases, and inner_ldr
    ldrs = ds.Dataset(args.ldrs)
    bases = np.load(args.bases)
    log.info(f'{bases.shape[1]} bases read from {args.bases}')
    if bases.shape[1] < args.n_ldrs:
        raise ValueError('the number of bases is less than the number of LDR')
    ldrs.data = ldrs.data[:, :args.n_ldrs]
    bases = bases[:, :args.n_ldrs]

    log.info(f'Read inner product of LDRs from {args.inner_ldr}')
    inner_ldr = np.load(args.inner_ldr)
    if inner_ldr.shape[0] < args.n_ldrs or inner_ldr.shape[1] < args.n_ldrs:
        raise ValueError(
            'the dimension of inner product of LDR is less than the number of LDR')
    inner_ldr = inner_ldr[:args.n_ldrs, :args.n_ldrs]
    log.info(f'Keep the top {args.n_ldrs} LDRs.\n')
    
    # keep
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
    else:
        keep_idvs = None
    log.info(f'{len(keep_idvs)} subjects in --keep.')

    # read covar and keep common subjects
    covar = ds.Covar(args.covar, args.cat_covar_list)
    common_idxs = ds.get_common_idxs(ldrs.data.index, covar.data.index, keep_idvs)
    covar.keep(common_idxs)
    covar.cat_covar_intercept()
    ldrs.keep(common_idxs)
    ldrs.to_single_index()
    ids = ldrs.data.index
    log.info(f'{len(ldrs)} subjects common in these files')
    
    covar.data = np.array(covar.data)
    ldrs.data = np.array(ldrs.data)
    n, p = covar.data.shape
    inner_covar_inv = np.linalg.inv(np.dot(covar.data.T, covar.data))  # (p, p)
    covar_ldrs = np.dot(covar.data.T, np.array(ldrs.data))  # (p, r)
    resid_ldr = ldrs.data - np.dot(covar.data, np.dot(inner_covar_inv, covar_ldrs)) # (n, r)
    var = np.sum(np.dot(bases, inner_ldr) * bases.T, axis=1) / (n - p)  # (N, )
    
    log.info(f'Save the null model to {args.out}_null_model.h5')
    with h5py.File(f'{args.out}_null_model.h5', 'w') as file:
        file.create_dataset('covar', data=covar.data)
        file.create_dataset('resid_ldr', data=resid_ldr)
        file.create_dataset('var', data=var)
        file.create_dataset('id', data=ids)
