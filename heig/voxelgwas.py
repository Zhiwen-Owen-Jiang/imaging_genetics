import numpy as np
import pandas as pd
from scipy.stats import chi2
from . import sumstats



def recover_se(bases, inner_ldr, n, ldr_beta, ztz_inv):
    bases = np.atleast_2d(bases)
    part1 = np.sum(np.dot(bases, inner_ldr) * bases, axis=1).reshape(1, -1) # 100*1
    part2 = np.dot(ldr_beta, bases.T) ** 2 # d*100
    se = np.sqrt(part1 * ztz_inv / n - part2 / n)
    return se


def check_input(args, log):
    ## required arguments
    if args.ldr_sumstats is None:
        raise ValueError('--ldr-sumstats is required.')
    if args.bases is None:
        raise ValueError('--bases is required.')
    if args.inner_ldr is None:
        raise ValueError('--inner-ldr is required.')
    
    ## optional arguments
    if args.range is not None and args.snp is not None:
        log.info('WARNING: --snp will be ignored if --range is provided.')
        args.snp = None
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0.')
    if args.voxel is not None and args.voxel < 0:
        raise ValueError('--voxel should be nonnegative.')
    if args.range is None and args.voxel is None and args.sig_thresh is None and args.snp is None:
        raise ValueError(('Generating all voxelwise summary statistics will require large disk memory. '
                          'Specify a p-value threshold by --sig-thresh to screen out insignificant results.'))
    if args.sig_thresh is not None and (args.sig_thresh <= 0 or args.sig_thresh >= 1):
        raise ValueError('--sig-thresh should be greater than 0 and less than 1.')

    ## process some arguments
    if args.range is not None:
        try:
            start, end = args.range.split(',')
            start_chr, start_pos = [int(x) for x in start.split(':')]
            end_chr, end_pos = [int(x) for x in end.split(':')]
        except:
            raise ValueError('--range should be in this format: 3:1000000,3:2000000.')
        if start_chr != end_chr:
            raise ValueError((f'The start chromosome is {start_chr} while the end chromosome is {end_chr}, '
                              'which is not allowed.'))
        if start_pos > end_pos:
            raise ValueError((f'The start position is {start_pos} while the end position is {end_pos}, '
                              'which is not allowed.'))
    else:
        start_chr, start_pos, end_chr, end_pos = None, None, None, None

    return start_chr, start_pos, end_pos



def run(args, log):
    target_chr, start_pos, end_pos = check_input(args, log)
    
    inner_ldr = np.load(args.inner_ldr)
    log.info(f'Read inner product of LDR from {args.inner_ldr}')
    bases = np.load(args.bases)
    log.info(f'{bases.shape[1]} bases read from {args.bases}.')
    ldr_gwas = sumstats.read_sumstats(args.ldr_sumstats)
    log.info(f'{ldr_gwas.snpinfo.shape[0]} SNPs read from LDR summary statistics {args.ldr_sumstats}.')

    if args.n_ldrs:
        if args.n_ldrs > ldr_gwas.beta.shape[1]:
            raise ValueError('--n-ldrs is greater than LDRs in summary statistics')
        else:
            ldr_gwas.beta = ldr_gwas.beta[:, :args.n_ldrs]
            ldr_gwas.se = ldr_gwas.se[:, :args.n_ldrs]
            bases = bases[:, :args.n_ldrs]
            inner_ldr = inner_ldr[:args.n_ldrs, :args.n_ldrs]
            log.info(f'Keep the top {args.n_ldrs} components.\n')

    ldr_n = np.array(ldr_gwas.snpinfo['N']).reshape(-1, 1)
    outpath = args.out

    if args.voxel is not None: 
        if args.voxel < bases.shape[0]:
            voxel_list = [args.voxel]
            outpath += f"_voxel{args.voxel}"
            log.info(f'Keep the voxel {args.voxel}.')
        else:
            raise ValueError('--voxel index out of range.')
    else:
        voxel_list = range(bases.shape[0])

    if target_chr:
        idx = ((ldr_gwas.snpinfo['POS'] > start_pos) & (ldr_gwas.snpinfo['POS'] < end_pos) & 
               (ldr_gwas.snpinfo['CHR'] == target_chr)).to_numpy()
        outpath += f"_chr{target_chr}_start{start_pos}_end{end_pos}.txt"
        log.info(f'Keep SNPs on chromosome {target_chr} from {start_pos} to {end_pos}.')
    elif args.snp:
        idx = (ldr_gwas.snpinfo['SNP'] == args.snp).to_numpy()
        outpath += f"_{args.snp}.txt"
        log.info(f'Keep SNP {args.snp}.')
    else:
        idx = ~ldr_gwas.snpinfo['SNP'].isna().to_numpy()
        outpath += ".txt"

    ldr_beta = ldr_gwas.beta[idx]
    ldr_se = ldr_gwas.se[idx]
    ldr_n = ldr_n[idx]
    snp_info = ldr_gwas.snpinfo.loc[idx]

    if args.sig_thresh:
        thresh_chisq = chi2.ppf(1 - args.sig_thresh, 1)
    else:
        thresh_chisq = 0 
    
    ztz_inv = np.mean((ldr_n * ldr_se ** 2 + ldr_beta ** 2) / np.diag(inner_ldr), axis=1)
    ztz_inv = ztz_inv.reshape(-1, 1)

    log.info(f"Recovering voxel-level GWAS results ...")
    is_first_write = True
    for i in voxel_list:
        if i % 100 == 1 and i > 1:
            log.info(f"Finished {i} voxels")
        voxel_beta = np.dot(ldr_beta, bases[i].T)
        voxel_se = recover_se(bases[i], inner_ldr, ldr_n, ldr_beta, ztz_inv).reshape(-1)
        voxel_z = voxel_beta / voxel_se
        sig_idxs = voxel_z ** 2 >= thresh_chisq

        if sig_idxs.any():
            sig_snps = snp_info.loc[sig_idxs].copy()
            sig_snps['BETA'] = voxel_beta[sig_idxs]
            sig_snps['SE'] = voxel_se[sig_idxs]
            sig_snps['Z'] = voxel_z[sig_idxs]
            sig_snps['P'] = chi2.sf(sig_snps['Z']**2, 1)
            sig_snps.insert(0, 'INDEX', [i] * np.sum(sig_idxs))
            
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
