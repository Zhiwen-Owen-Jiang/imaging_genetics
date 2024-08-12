import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from tqdm import tqdm
from heig import sumstats
import heig.input.dataset as ds


def recover_se(bases, inner_ldr, n, ldr_beta, ztz_inv):
    """
    Recovering standard errors for voxel-level genetic effects

    Parameters:
    ------------
    bases: a np.array of one functional base (N, r)
    inner_ldr: a np.array of inner product of LDRs (r, r)
    n: average sample size across all SNPs (1, )
    ldr_beta: LDR-level genetic effects (d, r)
    ztz_inv: a np.array of (Z_k'Z_k)^{-1} (d, 1)

    Returns:
    ---------
    se: a np.array of standard errors for voxel-level genetic effects (d, N)

    """
    bases = np.atleast_2d(bases)
    part1 = np.sum(np.dot(bases, inner_ldr) * bases,axis=1).reshape(1, -1)
    part2 = np.dot(ldr_beta, bases.T) ** 2
    se = np.sqrt(part1 * ztz_inv / n - part2 / n)

    return se


def check_input(args, log):
    # required arguments
    if args.ldr_sumstats is None:
        raise ValueError('--ldr-sumstats is required')
    if args.bases is None:
        raise ValueError('--bases is required')
    if args.inner_ldr is None:
        raise ValueError('--inner-ldr is required')

    # optional arguments
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0')
    if args.voxel is not None and args.voxel <= 0:
        raise ValueError('--voxel should be greater than 0 (one-based index)')
    if args.sig_thresh is not None and (args.sig_thresh <= 0 or args.sig_thresh >= 1):
        raise ValueError('--sig-thresh should be greater than 0 and less than 1')
    if args.range is None and args.voxel is None and args.sig_thresh is None and args.extract is None:
        log.info(('WARNING: generating all voxelwise summary statistics will require large disk space. '
                  'Specify a p-value threshold by --sig-thresh to screen out insignificant results.'))

    # required files must exist
    if not os.path.exists(f"{args.ldr_sumstats}.snpinfo"):
        raise FileNotFoundError(f"{args.ldr_sumstats}.snpinfo does not exist")
    if not os.path.exists(f"{args.ldr_sumstats}.sumstats"):
        raise FileNotFoundError(f"{args.ldr_sumstats}.sumstats does not exist")
    if not os.path.exists(args.bases):
        raise FileNotFoundError(f"{args.bases} does not exist")
    if not os.path.exists(args.inner_ldr):
        raise FileNotFoundError(f"{args.inner_ldr} does not exist")

    # process some arguments
    if args.range is not None:
        try:
            start, end = args.range.split(',')
            start_chr, start_pos = [int(x) for x in start.split(':')]
            end_chr, end_pos = [int(x) for x in end.split(':')]
        except:
            raise ValueError('--range should be in this format: <CHR>:<POS1>,<CHR>:<POS2>')
        if start_chr != end_chr:
            raise ValueError((f'starting with chromosome {start_chr} '
                              f'while ending with chromosome {end_chr} '
                              'is not allowed'))
        if start_pos > end_pos:
            raise ValueError((f'starting with {start_pos} '
                              f'while ending with position is {end_pos} '
                              'is not allowed'))
    else:
        start_chr, start_pos, end_chr, end_pos = None, None, None, None

    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
    else:
        keep_snps = None

    return start_chr, start_pos, end_pos, keep_snps


def run(args, log):
    # checking input
    target_chr, start_pos, end_pos, keep_snps = check_input(args, log)

    # reading data
    inner_ldr = np.load(args.inner_ldr)
    log.info(f'Read inner product of LDRs from {args.inner_ldr}')
    bases = np.load(args.bases)
    log.info(f'{bases.shape[1]} bases read from {args.bases}')
    ldr_gwas = sumstats.read_sumstats(args.ldr_sumstats)
    log.info(f'{ldr_gwas.snpinfo.shape[0]} SNPs read from LDR summary statistics {args.ldr_sumstats}')

    # LDR subsetting
    if args.n_ldrs:
        if args.n_ldrs > ldr_gwas.beta.shape[1] or args.n_ldrs > bases.shape[1] or args.n_ldrs > inner_ldr.shape[0]:
            log.info('WARNING: --n-ldrs is greater than the maximum #LDRs. Use all LDRs.')
        else:
            ldr_gwas.beta = ldr_gwas.beta[:, :args.n_ldrs]
            ldr_gwas.se = ldr_gwas.se[:, :args.n_ldrs]
            bases = bases[:, :args.n_ldrs]
            inner_ldr = inner_ldr[:args.n_ldrs, :args.n_ldrs]

    if bases.shape[1] != ldr_gwas.beta.shape[1] or bases.shape[1] != inner_ldr.shape[0]:
        raise ValueError('dimension mismatch for --bases, --inner-ldr, and --ldr-sumstats. Try to use --n-ldrs')
    log.info(f'Keep the top {bases.shape[1]} components.\n')

    # getting the outpath and SNP list
    outpath = args.out
    if args.voxel is not None:
        if args.voxel <= bases.shape[0]:  # one-based index
            voxel_list = [args.voxel - 1]
            outpath += f"_voxel{args.voxel}"
            log.info(f'Keep the voxel {args.voxel}.')
        else:
            raise ValueError('--voxel index (one-based) out of range')
    else:
        voxel_list = range(bases.shape[0])

    if target_chr:
        idx = ((ldr_gwas.snpinfo['POS'] > start_pos) & (ldr_gwas.snpinfo['POS'] < end_pos) &
               (ldr_gwas.snpinfo['CHR'] == target_chr)).to_numpy()
        outpath += f"_chr{target_chr}_start{start_pos}_end{end_pos}.txt"
        log.info(f'{np.sum(idx)} SNP(s) on chromosome {target_chr} from {start_pos} to {end_pos}.')
    else:
        idx = ~ldr_gwas.snpinfo['SNP'].isna().to_numpy()
        outpath += ".txt"
        log.info(f'{np.sum(idx)} SNP(s) in total.')
    
    if keep_snps is not None:
        idx_keep_snps = (ldr_gwas.snpinfo['SNP'].isin(keep_snps['SNP'])).to_numpy()
        idx = idx & idx_keep_snps
        log.info(f"Keep {len(keep_snps['SNP'])} SNP(s) from --extract.")

    # extracting SNPs
    ldr_beta = ldr_gwas.beta[idx]
    ldr_se = ldr_gwas.se[idx]
    ldr_n = np.array(ldr_gwas.snpinfo['N']).reshape(-1, 1)
    ldr_n = ldr_n[idx]
    # ldr_n = np.max(ldr_n) # to get the image sample size
    snp_info = ldr_gwas.snpinfo.loc[idx]

    # getting threshold
    if args.sig_thresh:
        thresh_chisq = chi2.ppf(1 - args.sig_thresh, 1)
    else:
        thresh_chisq = 0

    # estimating (Z_k'Z_k)^{-1}
    ztz_inv = np.mean((ldr_n * ldr_se ** 2 + ldr_beta ** 2) / np.diag(inner_ldr), axis=1)
    ztz_inv = ztz_inv.reshape(-1, 1)

    # doing analysis
    log.info(f"Recovering voxel-level GWAS results ...")
    is_first_write = True
    for i in tqdm(voxel_list, desc=f"Doing GWAS for {len(voxel_list)} voxels"):
        voxel_beta = np.dot(ldr_beta, bases[i].T)
        voxel_se = recover_se(bases[i], inner_ldr, ldr_n, ldr_beta, ztz_inv).reshape(-1)
        voxel_z = voxel_beta / voxel_se
        sig_idxs = voxel_z ** 2 >= thresh_chisq

        if sig_idxs.any():
            sig_snps = snp_info.loc[sig_idxs].copy()
            sig_snps['BETA'] = voxel_beta[sig_idxs]
            sig_snps['SE'] = voxel_se[sig_idxs]
            sig_snps['Z'] = voxel_z[sig_idxs]
            sig_snps['P'] = chi2.sf(sig_snps['Z'] ** 2, 1)
            sig_snps.insert(0, 'INDEX', [i+1] * np.sum(sig_idxs))

            if is_first_write:
                sig_snps_output = sig_snps.to_csv(sep='\t', header=True, na_rep='NA', index=None,
                                                  float_format='%.5e')
                is_first_write = False
                with open(outpath, 'w') as file:
                    file.write(sig_snps_output)
            else:
                sig_snps_output = sig_snps.to_csv(sep='\t', header=False, na_rep='NA', index=None,
                                                  float_format='%.5e')
                with open(outpath, 'a') as file:
                    file.write(sig_snps_output)

    log.info(f"Save the output to {outpath}")
