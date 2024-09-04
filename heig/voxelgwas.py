import os
import numpy as np
from scipy.stats import chi2
from tqdm import tqdm
from heig import sumstats
import heig.input.dataset as ds



class VGWAS:
    def __init__(self, bases, inner_ldr, ldr_gwas, snp_idxs, n):
        """
        Parameters:
        ------------
        bases: a np.array of bases (N, r)
        inner_ldr: a np.array of inner product of LDRs (r, r)
        ldr_gwas: a GWAS instance
        snp_idxs: numerical indices of SNPs to extract (d, )
        n: sample sizes of SNPs (d, 1)
        
        """
        self.bases = bases
        self.inner_ldr = inner_ldr
        self.ldr_gwas = ldr_gwas
        self.ldr_idxs = list(range(ldr_gwas.n_gwas))
        self.snp_idxs = snp_idxs
        self.n = n
        self.ztz_inv = self._compute_ztz_inv() # (d, 1)

    def _compute_ztz_inv(self):
        """
        Computing (Z'Z)^{-1} from summary statistics

        Returns:
        ---------
        ztz_inv: a np.array of (Z'Z)^{-1} (d, 1)
        
        """
        diag_inner_ldr = np.diag(self.inner_ldr)
        ztz_inv = np.zeros(np.sum(self.snp_idxs))
        n_ldrs = self.bases.shape[1]

        i = 0
        data_reader = self.ldr_gwas.data_reader(['beta', 'se'], self.ldr_idxs, self.snp_idxs)
        for ldr_beta_batch, ldr_se_batch in data_reader:
            batch_size = ldr_beta_batch.shape[1]
            ztz_inv += np.sum((self.n * ldr_se_batch ** 2 + ldr_beta_batch ** 2) / diag_inner_ldr[i: i+batch_size], axis=1)
            i += batch_size
        ztz_inv /= n_ldrs
        ztz_inv = ztz_inv.reshape(-1, 1)
        
        return ztz_inv
    
    def recover_beta(self, voxel_idxs):
        """
        Recovering voxel beta

        Parameters:
        ------------
        voxel_idxs: a list of voxel idxs (q)

        Returns:
        ---------
        voxel_beta: a np.array of voxel beta (d, q)
        
        """
        voxel_beta = np.zeros((np.sum(self.snp_idxs), len(voxel_idxs)))
        data_reader = self.ldr_gwas.data_reader(['beta'], self.ldr_idxs, self.snp_idxs)
        base = self.bases[voxel_idxs] # (q, r)

        i = 0
        for ldr_beta_batch in data_reader:
            ldr_beta_batch = ldr_beta_batch[0]
            batch_size = ldr_beta_batch.shape[1]
            voxel_beta += np.dot(ldr_beta_batch, base[:, i: i+batch_size].T)
            i += batch_size

        return voxel_beta
    
    def recover_se(self, voxel_idxs, voxel_beta):
        """
        Recovering standard errors for voxel-level genetic effects

        Parameters:
        ------------
        voxel_idx: a list of voxel idxs (q)
        voxel_beta: a np.array of voxel beta (d, q)

        Returns:
        ---------
        voxel_se: a np.array of standard errors for voxel-level genetic effects (d, )

        """
        base = np.atleast_2d(self.bases[voxel_idxs]) # (q, r)
        part1 = np.sum(np.dot(base, self.inner_ldr) * base, axis=1) # (q, )
        # part2 = (voxel_beta ** 2) # (d, q)
        voxel_se = np.sqrt(part1 * self.ztz_inv / self.n - voxel_beta ** 2 / self.n)

        return voxel_se


def voxel_reader(n_snps, voxel_list):
    n_voxels = len(voxel_list)
    memory_use = n_snps * n_voxels * np.dtype(np.float32).itemsize / (1024 ** 3)
    if memory_use <= 1:
        batch_size = n_voxels
    else:
        batch_size = int(n_voxels / memory_use)

    for i in range(0, n_voxels, batch_size):
        yield voxel_list[i: i+batch_size]


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

    try:
        ldr_gwas = sumstats.read_sumstats(args.ldr_sumstats)
        log.info(f'{ldr_gwas.n_snps} SNPs read from LDR summary statistics {args.ldr_sumstats}')

        # LDR subsetting
        if args.n_ldrs:
            if args.n_ldrs > ldr_gwas.n_gwas or args.n_ldrs > bases.shape[1] or args.n_ldrs > inner_ldr.shape[0]:
                log.info('WARNING: --n-ldrs is greater than the maximum #LDRs. Use all LDRs.')
            else:
                ldr_gwas.n_gwas = args.n_ldrs
                bases = bases[:, :args.n_ldrs]
                inner_ldr = inner_ldr[:args.n_ldrs, :args.n_ldrs]

        if bases.shape[1] != ldr_gwas.n_gwas or bases.shape[1] != inner_ldr.shape[0]:
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
            voxel_list = list(range(bases.shape[0]))

        if target_chr:
            snp_idxs = ((ldr_gwas.snpinfo['POS'] > start_pos) & (ldr_gwas.snpinfo['POS'] < end_pos) &
                (ldr_gwas.snpinfo['CHR'] == target_chr)).to_numpy()
            outpath += f"_chr{target_chr}_start{start_pos}_end{end_pos}.txt"
            log.info(f'{np.sum(snp_idxs)} SNP(s) on chromosome {target_chr} from {start_pos} to {end_pos}.')
        else:
            snp_idxs = ~ldr_gwas.snpinfo['SNP'].isna().to_numpy()
            outpath += ".txt"
            log.info(f'{np.sum(snp_idxs)} SNP(s) in total.')
        
        if keep_snps is not None:
            idx_keep_snps = (ldr_gwas.snpinfo['SNP'].isin(keep_snps['SNP'])).to_numpy()
            snp_idxs = snp_idxs & idx_keep_snps
            log.info(f"Keep {len(keep_snps['SNP'])} SNP(s) from --extract.")

        # extracting SNPs
        ldr_n = np.array(ldr_gwas.snpinfo['N']).reshape(-1, 1)
        ldr_n = ldr_n[snp_idxs]
        snp_info = ldr_gwas.snpinfo.loc[snp_idxs]

        # getting threshold
        if args.sig_thresh:
            thresh_chisq = chi2.ppf(1 - args.sig_thresh, 1)
        else:
            thresh_chisq = 0

        # doing analysis
        log.info(f"Recovering voxel-level GWAS results ...")
        vgwas = VGWAS(bases, inner_ldr, ldr_gwas, snp_idxs, ldr_n)
        is_first_write = True
        for voxel_idxs in tqdm(voxel_reader(np.sum(snp_idxs), voxel_list), desc=f"Doing GWAS for {len(voxel_list)} voxels"):
            voxel_beta = vgwas.recover_beta(voxel_idxs)
            voxel_se = vgwas.recover_se(voxel_idxs, voxel_beta)
            voxel_z = voxel_beta / voxel_se
            all_sig_idxs = voxel_z ** 2 >= thresh_chisq
            all_sig_idxs_voxel = all_sig_idxs.any(axis=0)

            for i, voxel_idx in enumerate(voxel_idxs):
                if all_sig_idxs_voxel[i]:
                    sig_idxs = all_sig_idxs[:, i]
                    sig_snps = snp_info.loc[sig_idxs].copy()
                    sig_snps['BETA'] = voxel_beta[sig_idxs, i]
                    sig_snps['SE'] = voxel_se[sig_idxs, i]
                    sig_snps['Z'] = voxel_z[sig_idxs, i]
                    sig_snps['P'] = chi2.sf(sig_snps['Z'] ** 2, 1)
                    sig_snps.insert(0, 'INDEX', [voxel_idx+1] * np.sum(sig_idxs))

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

    finally:
        ldr_gwas.file.close()
