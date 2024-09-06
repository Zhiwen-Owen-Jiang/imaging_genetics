import os
import h5py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import chi2
from heig import sumstats
import heig.input.dataset as ds
from heig.ldmatrix import LDmatrix
from heig.ldsc import LDSC



def check_input(args, log):
    # required arguments
    if args.ldr_sumstats is None:
        raise ValueError('--ldr-sumstats is required')
    if args.bases is None:
        raise ValueError('--bases is required')
    if args.inner_ldr is None:
        raise ValueError('--inner-ldr is required')
    if args.ld_inv is None:
        raise ValueError('--ld-inv is required')
    if args.ld is None:
        raise ValueError('--ld is required')

    if not os.path.exists(f"{args.ldr_sumstats}.snpinfo"):
        raise FileNotFoundError(f"{args.ldr_sumstats}.snpinfo does not exist")
    if not os.path.exists(f"{args.ldr_sumstats}.sumstats"):
        raise FileNotFoundError(
            f"{args.ldr_sumstats}.sumstats does not exist")
    if not os.path.exists(args.bases):
        raise FileNotFoundError(f"{args.bases} does not exist")
    if not os.path.exists(args.inner_ldr):
        raise FileNotFoundError(f"{args.inner_ldr} does not exist")
    if args.overlap and not args.y2_sumstats:
        log.info('WARNING: ignore --overlap as --y2-sumstats is not specified.')
    if args.y2_sumstats is not None:
        if not os.path.exists(f"{args.y2_sumstats}.snpinfo"):
            raise FileNotFoundError(
                f"{args.y2_sumstats}.snpinfo does not exist")
        if not os.path.exists(f"{args.y2_sumstats}.sumstats"):
            raise FileNotFoundError(
                f"{args.y2_sumstats}.sumstats does not exist")


def get_common_snps(*snp_list):
    """
    Extracting common snps from multiple snp lists

    Parameters:
    ------------
    snp_list: a list of snp lists

    Returns:
    ---------
    common_snps: a list of common snps

    """
    n_snp_list = len(snp_list)
    if n_snp_list == 0:
        raise ValueError('no SNP list provided')

    common_snps = None
    for i in range(len(snp_list)):
        if hasattr(snp_list[i], 'ldinfo'):
            snp = snp_list[i].ldinfo[['SNP', 'A1', 'A2']]
            snp = snp.rename({'A1': f'A1_{i}', 'A2': f'A2_{i}'}, axis=1)
        elif hasattr(snp_list[i], 'snpinfo'):
            snp = snp_list[i].snpinfo[['SNP', 'A1', 'A2']]
            snp = snp.rename({'A1': f'A1_{i}', 'A2': f'A2_{i}'}, axis=1)
        elif hasattr(snp_list[i], 'SNP'):
            snp = snp_list[i]['SNP']
        if not isinstance(common_snps, pd.DataFrame):
            common_snps = snp.copy()
        else:
            common_snps = common_snps.merge(snp, on='SNP')

    if common_snps is None:
        raise ValueError('all the input snp lists are None or do not have a SNP column')

    common_snps.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
    if len(common_snps) == 0:
        raise ValueError('no common SNPs exist')

    matched_alleles_set = common_snps[[col for col in common_snps.columns
                                       if col.startswith('A')]].apply(lambda x: len(set(x)) == 2, axis=1)
    common_snps = common_snps.loc[matched_alleles_set]

    return common_snps['SNP']


def keep_ldrs(n_ldrs, bases, inner_ldr, ldr_gwas):
    """
    Extracting LDRs
    
    """
    if bases.shape[1] < n_ldrs:
        raise ValueError('the number of bases is less than --n-ldrs')
    if inner_ldr.shape[0] < n_ldrs:
        raise ValueError('the dimension of inner product of LDR is less than --n-ldrs')
    if ldr_gwas.n_gwas < n_ldrs:
        raise ValueError('LDRs in summary statistics is less than --n-ldrs')
    bases = bases[:, :n_ldrs]
    inner_ldr = inner_ldr[:n_ldrs, :n_ldrs]
    ldr_gwas.n_gwas = n_ldrs

    return bases, inner_ldr, ldr_gwas


class Estimation(ABC):
    def __init__(self, ldr_gwas, ld, ld_inv, bases, inner_ldr):
        """
        An abstract class for estimating heritability and genetic correlation

        Parameters:
        ------------
        ldr_gwas: a GWAS instance
        ld: a LDmatrix instance
        ld_inv: a LDmatrix instance
        bases: a np.array of bases (N, r)
        inner_ldr: a np.array of inner product of LDRs (r, r)

        """
        self.ldr_gwas = ldr_gwas
        self.n = ldr_gwas.snpinfo['N'].values.reshape(-1, 1)
        self.N = bases.shape[0]
        self.nbar = np.mean(self.n)
        self.ld = ld
        self.ld_inv = ld_inv
        self.r = ldr_gwas.n_gwas
        self.bases = bases
        self.inner_ldr = inner_ldr
        self.block_ranges = ld.block_ranges
        self.sigmaX_var = np.sum(np.dot(bases, inner_ldr) * bases, axis=1) / self.nbar

    def _ldr_sumstats_reader(self, start, end, all_ldrs=True, normal=True):
        """
        Reading LDRs sumstats from HDF5 file and preprocessing:
        0. selecting a subset of LDRs
        1. selecting SNPs, reading beta, se into memory, getting z
        2. flipping +/-
        3. normalization (optional)
        4. yield

        Parameters:
        ------------
        start: start index of a LD block
        end: end index of a LD block
        all_ldrs: if extracting all LDR sumstats
        normal: if doing normalization

        Returns:
        ---------
        preprocessed z scores
        
        """
        ldr_idxs = list(range(self.ldr_gwas.n_gwas))
        block_snp_idxs = self.ldr_gwas.snp_idxs[start: end]
        block_change_sign = self.ldr_gwas.change_sign[start: end]
        block_n = self.n[start: end]
        data_reader = self.ldr_gwas.data_reader(['beta', 'se'], 
                                                ldr_idxs, 
                                                block_snp_idxs,
                                                all_gwas=all_ldrs)
            
        sqrt_diag_inner_ldr = np.sqrt(np.diag(self.inner_ldr))
        for z, se in data_reader:
            z = z / se
            z[block_change_sign] = -1 * z[block_change_sign]
            if normal:
                z = z * sqrt_diag_inner_ldr / block_n
            yield z

    def _get_heri_se(self, heri, d, n):
        """
        Estimating standard error of heritability estimates

        Parameters:
        ------------
        heri: an N by 1 vector of heritability estimates
        d: number of SNPs, or Tr(R\Omega)  
        n: sample size

        Returns:
        ---------
        An N by 1 vector of standard error estimates

        """
        part1 = 2 * d / n ** 2
        part2 = 2 * heri / n
        part3 = 2 * heri * (1 - heri) / n
        res = part1 + part2 + part3
        res[np.abs(res) < 10 ** -10] = 0
        return np.sqrt(res)
    
    @staticmethod
    def _qc(est, est_se, est_min, est_max, est_se_min, est_se_max):
        invalid_est = ((est > est_max) | (est < est_min) | 
                       (est_se > est_se_max) | (est_se < est_se_min))
        est[invalid_est] = np.nan
        est_se[invalid_est] = np.nan
        return est, est_se

    @abstractmethod
    def _block_wise_estimate(self):
        pass


class OneSample(Estimation):
    def __init__(self, ldr_gwas, ld, ld_inv, bases, inner_ldr):
        super().__init__(ldr_gwas, ld, ld_inv, bases, inner_ldr)

        self.ld_rank = 0
        self.ldr_gene_cov = np.zeros((self.r, self.r))

        for (start, end), ld_block, ld_inv_block in zip(self.block_ranges, self.ld.data, self.ld_inv.data):
            ld_block_rank, block_gene_cov = self._block_wise_estimate(
                start, end, ld_block, ld_inv_block
            )
            self.ld_rank += ld_block_rank
            self.ldr_gene_cov += block_gene_cov
        self.gene_var = np.sum(np.dot(self.bases, self.ldr_gene_cov) * self.bases, axis=1)
        self.heri = self.gene_var / self.sigmaX_var
        self.heri_se = self._get_heri_se(self.heri, self.ld_rank, self.nbar)
        self.heri, self.heri_se = self._qc(self.heri, self.heri_se, 0, 1, 0, 1)

    def _block_wise_estimate(self, start, end, ld_block, ld_block_inv):
        """
        block_ld: eigenvectors * sqrt(eigenvalues)
        block_ld_inv: eigenvectors * sqrt(eigenvalues ** -1)

        """
        block_ld_ld_inv = np.dot(ld_block.T, ld_block_inv)
        ld_block_rank = np.sum(block_ld_ld_inv * block_ld_ld_inv)
        block_z = next(self._ldr_sumstats_reader(start, end))
        z_mat_block_ld_inv = np.dot(block_z.T, ld_block_inv) # (r, ld_size)
        block_gene_cov = (np.dot(z_mat_block_ld_inv, z_mat_block_ld_inv.T) -
                          ld_block_rank * self.inner_ldr / self.nbar ** 2) # (r, r)

        return ld_block_rank, block_gene_cov

    def get_gene_cor_se(self, out_dir):
        """
        Computing genetic correlation and its se by block
        Saving to a HDF5 file
        
        """
        block_size = 100
        mean_gene_cor = 0
        min_gene_cor = 1
        mean_gene_cor_se = 0
        with h5py.File(f'{out_dir}_gc.h5', 'w') as file:
            gc = file.create_dataset('gc', shape=(self.N, self.N), dtype='float32')
            se = file.create_dataset('se', shape=(self.N, self.N), dtype='float32')
            for i in range(0, self.N, block_size):
                start = i
                end = i + block_size
                gene_cor, gene_cor_se = self._get_gene_cor_se_block(start, end)
                gc[start: end] = gene_cor
                se[start: end] = gene_cor_se
                mean_gene_cor += np.nansum(gene_cor)
                mean_gene_cor_se += np.nansum(gene_cor_se)
                min_gene_cor = np.min((min_gene_cor, np.nanmin(gene_cor)))
        mean_gene_cor /= self.N**2
        mean_gene_cor_se /= self.N**2

        return mean_gene_cor, min_gene_cor, mean_gene_cor_se
    
    def _get_gene_cor_se_block(self, start, end):
        """
        Computing genetic correlation and se for a block

        """
        gene_cov = np.dot(np.dot(self.bases[start: end], self.ldr_gene_cov), self.bases.T)
        gene_cov[gene_cov == 0] = 0.01
        gene_cor = gene_cov / np.sqrt(np.outer(self.gene_var[start: end], self.gene_var))
        gene_cor2 = gene_cor * gene_cor
        part1 = 1 - gene_cor2

        part2 = np.dot(np.dot(self.bases[start: end], self.inner_ldr), self.bases.T) / self.nbar / gene_cov
        part2 -= 1
        del gene_cov

        inv_heri1 = np.repeat(1 / self.heri[start: end] - 1, self.N).reshape(-1, self.N)
        inv_heri2 = np.repeat(1 / self.heri - 1, gene_cor.shape[0]).reshape(self.N, -1).T
        part3 = inv_heri1 + inv_heri2
        temp1 = inv_heri1 - part2
        temp2 = inv_heri2 - part2
        temp3 = inv_heri1 * inv_heri2
        del inv_heri1, inv_heri2

        gene_cor_se = np.zeros((gene_cor.shape[0], self.N))
        n = self.nbar
        d = self.ld_rank
        gene_cor_se += (4 / n + d / n**2) * part1 * part1
        gene_cor_se += (1 / n + d / n**2) * part1 * part3
        gene_cor_se -= (1 / n + d / n**2) * part1 * 2 * gene_cor2 * part2
        gene_cor_se += d / n**2 / 2 * gene_cor2 * temp1 * temp1
        gene_cor_se += d / n**2 / 2 * gene_cor2 * temp2 * temp2
        gene_cor_se += d / n**2 * gene_cor2 * gene_cor2 * part2 * part2
        gene_cor_se += d / n**2 * temp3
        gene_cor_se -= d / n**2 * gene_cor2 * part2 * part3
        
        gene_cor_se[np.abs(gene_cor_se) < 10 ** -10] = 0
        gene_cor, gene_cor_se = self._qc(gene_cor, gene_cor_se, -1, 1, 0, 1)

        return gene_cor, np.sqrt(gene_cor_se)


class TwoSample(Estimation):
    def __init__(self, ldr_gwas, ld, ld_inv, bases, inner_ldr,
                 y2_gwas, overlap=False):
        super().__init__(ldr_gwas, ld, ld_inv, bases, inner_ldr)
        self.y2_gwas = y2_gwas
        self.n2 = y2_gwas.snpinfo['N'].values.reshape(-1, 1)
        self.n2bar = np.mean(self.n2)

        y2_block_gene_cov = np.zeros(len(self.block_ranges))
        ldr_y2_block_gene_cov_part1 = np.zeros((len(self.block_ranges), self.r))
        self.ld_block_rank = np.zeros(len(self.block_ranges))
        self.ldr_block_gene_cov = np.zeros((len(self.ld_block_rank), self.r, self.r))

        for i, ((start, end), ld_block, ld_inv_block) in enumerate(zip(self.block_ranges, self.ld.data, self.ld_inv.data)):
            ld_block_rank, block_gene_var_y2, block_gene_cov, block_gene_cov_y2 = self._block_wise_estimate(
                start, end, ld_block, ld_inv_block)
            self.ld_block_rank[i] = ld_block_rank
            self.ldr_block_gene_cov[i, :, :] = block_gene_cov
            y2_block_gene_cov[i] = block_gene_var_y2
            ldr_y2_block_gene_cov_part1[i, :] = block_gene_cov_y2

        # since the sum stats have been normalized
        self.ld_rank = np.sum(self.ld_block_rank)
        self.ldr_gene_cov = np.sum(self.ldr_block_gene_cov, axis=0)
        self.y2_heri = np.atleast_1d(np.sum(y2_block_gene_cov))
        ldr_y2_gene_cov_part1 = np.sum(ldr_y2_block_gene_cov_part1, axis=0)
        
        self.heri = (np.sum(np.dot(self.bases, self.ldr_gene_cov)
                     * self.bases, axis=1) / self.sigmaX_var)
        self.heri_se = self._get_heri_se(self.heri, self.ld_rank, self.nbar)
        self.heri, self.heri_se = self._qc(self.heri, self.heri_se, 0, 1, 0, 1)

        if not overlap:
            self.gene_cor_y2 = np.squeeze(self._get_gene_cor_y2(ldr_y2_gene_cov_part1,
                                                                self.heri,
                                                                self.y2_heri))
            self.gene_cor_y2_se = self._get_gene_cor_se(self.heri, self.y2_heri,
                                                        self.gene_cor_y2,
                                                        self.ld_rank, self.nbar,
                                                        self.n2bar)
        else:
            merged_blocks = ld.merge_blocks()
            ldscore = ld.ldinfo['ldscore'].values
            n_merged_blocks = len(merged_blocks)

            # compute left-one-block-out heritability
            ldr_lobo_gene_cov = self._lobo_estimate(self.ldr_gene_cov,
                                                    self.ldr_block_gene_cov,
                                                    merged_blocks)
            y2_lobo_heri = self._lobo_estimate(self.y2_heri, y2_block_gene_cov, merged_blocks)
            temp = np.matmul(np.swapaxes(ldr_lobo_gene_cov, 1, 2), bases.T).swapaxes(1, 2)
            temp = np.sum(temp * np.expand_dims(bases, 0), axis=2)
            image_lobo_heri = temp / self.sigmaX_var

            # compute left-one-block-out cross-trait LDSC intercept
            self.ldr_heri = np.diag(self.ldr_gene_cov) / np.diag(self.inner_ldr) * self.nbar
            ldr_sumstats_reader = self._ldr_sumstats_reader(0, self.ldr_gwas.n_snps, 
                                                            all_ldrs=False, normal=False)
            y2_sumstats_reader = self._y2_sumstats_reader(0, self.y2_gwas.n_snps, normal=False)
            ldsc_intercept = LDSC(ldr_sumstats_reader, y2_sumstats_reader, ldscore, self.ldr_heri, self.y2_heri,
                                  self.n, self.n2, self.ld_rank, self.block_ranges,
                                  merged_blocks)

            # compute left-one-block-out genetic correlation
            ldr_y2_lobo_gene_cov_part1 = self._lobo_estimate(ldr_y2_gene_cov_part1,
                                                             ldr_y2_block_gene_cov_part1,
                                                             merged_blocks)
            ld_rank_lobo = self._lobo_estimate(self.ld_rank,
                                               self.ld_block_rank,
                                               merged_blocks)
            lobo_gene_cor = self._get_gene_cor_ldsc(ldr_y2_lobo_gene_cov_part1,
                                                    ld_rank_lobo,
                                                    ldsc_intercept.lobo_ldsc,
                                                    image_lobo_heri,
                                                    y2_lobo_heri)  # n_blocks * N

            # compute genetic correlation using all blocks
            image_y2_gene_cor = self._get_gene_cor_ldsc(ldr_y2_gene_cov_part1,
                                                        self.ld_rank,
                                                        ldsc_intercept.total_ldsc,
                                                        self.heri,
                                                        self.y2_heri)  # 1 * N

            # compute jackknite estimate of genetic correlation and se
            self.gene_cor_y2, self.gene_cor_y2_se = self._jackknife(image_y2_gene_cor,
                                                                    lobo_gene_cor,
                                                                    n_merged_blocks)

        self.y2_heri_se = self._get_heri_se(self.y2_heri, self.ld_rank, self.n2bar)
        self.gene_cor_y2, self.gene_cor_y2_se = self._qc(self.gene_cor_y2, self.gene_cor_y2_se, 
                                                         -1, 1, 0, 1)
        self.y2_heri, self.y2_heri_se = self._qc(self.y2_heri, self.y2_heri_se, 
                                                 0, 1, 0, 1)

    def _y2_sumstats_reader(self, start, end, normal=True):
        """
        Reading LDRs summstats from HDF5 file and preprocessing:
        1. selecting SNPs, reading z
        2. flipping +/-
        3. normalization
        4. yield

        Parameters:
        ------------
        start: start index of a LD block
        end: end index of a LD block
        normal: if do normalization

        Returns:
        ---------
        preprocessed z scores
        
        """
        y2_idxs = list(range(self.y2_gwas.n_gwas))
        block_snp_idxs = self.y2_gwas.snp_idxs[start: end]
        block_change_sign = self.y2_gwas.change_sign[start: end]
        block_n = self.n2[start: end]
        data_reader = self.y2_gwas.data_reader(['z'],
                                            y2_idxs, 
                                            block_snp_idxs,
                                            all_gwas=False)
        
        for z in data_reader:
            z = z[0]
            z[block_change_sign] = -1 * z[block_change_sign]
            if normal:
                z = z / np.sqrt(block_n)
            yield z

    def _block_wise_estimate(self, start, end, ld_block, ld_block_inv):
        """
        ld_block: eigenvectors * sqrt(eigenvalues)
        ld_block_inv: eigenvectors * sqrt(eigenvalues ** -1)

        """
        ld_block_ld_inv = np.dot(ld_block.T, ld_block_inv)
        ld_block_rank = np.sum(ld_block_ld_inv * ld_block_ld_inv)

        block_z = next(self._ldr_sumstats_reader(start, end))
        z_mat_ld_block_inv = np.dot(block_z.T, ld_block_inv)
        block_y2z = next(self._y2_sumstats_reader(start, end))
        y2_ld_block_inv = np.dot(block_y2z.T, ld_block_inv)

        block_gene_var_y2 = np.dot(y2_ld_block_inv, y2_ld_block_inv.T) - ld_block_rank / self.n2bar
        block_gene_cov_y2 = np.squeeze(np.dot(z_mat_ld_block_inv, y2_ld_block_inv.T))
        block_gene_cov = (np.dot(z_mat_ld_block_inv, z_mat_ld_block_inv.T) -
                          ld_block_rank * self.inner_ldr / self.nbar ** 2)

        return ld_block_rank, block_gene_var_y2, block_gene_cov, block_gene_cov_y2

    def _get_gene_cor_ldsc(self, part1, ld_rank, ldsc, heri1, heri2):
        ldr_gene_cov = (part1 - ld_rank.reshape(-1, 1) * ldsc *
                        np.sqrt(np.diagonal(self.inner_ldr)) / self.nbar / np.sqrt(self.n2bar))
        gene_cor = self._get_gene_cor_y2(ldr_gene_cov.T, heri1, heri2)

        return gene_cor

    def _get_gene_cor_y2(self, inner_part, heri1, heri2):
        bases_inner_part = np.dot(self.bases, inner_part).reshape(self.bases.shape[0], -1)
        gene_cov_y2 = bases_inner_part / np.sqrt(self.sigmaX_var).reshape(-1, 1)
        gene_cor_y2 = gene_cov_y2.T / np.sqrt(heri1 * heri2)

        return gene_cor_y2

    def _jackknife(self, total, lobo, n_blocks):
        """
        Jackknite estimator

        Parameters:
        ------------
        total: the estimate using all blocks
        lobo: an n_blocks by N matrix of lobo estimates
        n_blocks: the number of blocks

        Returns:
        ---------
        estimate: an N by 1 vector of jackknife estimates
        se: an N by 1 vector of jackknife se estimates

        """
        mean_lobo = np.mean(lobo, axis=0)
        estimate = np.squeeze(n_blocks * total - (n_blocks - 1) * mean_lobo)
        se = np.squeeze(np.sqrt((n_blocks - 1) / n_blocks * 
                                np.sum((lobo - mean_lobo) ** 2, axis=0)))

        return estimate, se

    def _lobo_estimate(self, total_est, block_est, merged_blocks):
        lobo_est = []

        for blocks in merged_blocks:
            lobo_est_i = total_est.copy()
            for ii in blocks:
                lobo_est_i -= block_est[ii]
            lobo_est.append(lobo_est_i)

        return np.array(lobo_est)

    def _get_gene_cor_se(self, heri, heri_y2, gene_cor_y2, d, n1, n2):
        """
        Estimating standard error of two-sample genetic correlation estimates
        without sample overlap

        Parameters:
        ------------
        heri: an N by 1 vector of heritability estimates of images
        heri_y2: an 1 by 1 number of heritability estimate of a single trait
        gene_cor_y2: an N by 1 vector of genetic correlation estimates 
                    between images and a single trait
        d: number of SNPs, or Tr(R\Omega)  
        n1: sample size of images
        n2: sample size of the single trait

        Returns:
        ---------
        An N by N matrix of standard error estimates
        This estimator assumes no sample overlap

        """
        gene_cor_y2sq = gene_cor_y2 * gene_cor_y2
        gene_cor_y2sq1 = 1 - gene_cor_y2sq

        var = np.zeros(self.N)
        var += gene_cor_y2sq / (2 * heri * heri) * d / n1 ** 2
        var += gene_cor_y2sq / (2 * heri_y2 * heri_y2) * d / n2 ** 2
        var += 1 / (heri * heri_y2) * d / (n1 * n2)
        var += gene_cor_y2sq1 / (heri * n1)
        var += gene_cor_y2sq1 / (heri_y2 * n2)

        return np.sqrt(var)


def format_heri(heri, heri_se, log):
    log.info('Removed out-of-bound results (if any)\n')
    chisq = (heri / heri_se) ** 2
    pv = chi2.sf(chisq, 1)
    data = {'INDEX': range(1, len(heri) + 1), 'H2': heri, 'SE': heri_se,
            'CHISQ': chisq, 'P': pv}
    output = pd.DataFrame(data)

    return output


def format_gene_cor_y2(heri, heri_se, gene_cor, gene_cor_se, log):
    log.info('Removed out-of-bound results (if any)\n')
    heri_chisq = (heri / heri_se) ** 2
    heri_pv = chi2.sf(heri_chisq, 1)
    gene_cor_chisq = (gene_cor / gene_cor_se) ** 2
    gene_cor_pv = chi2.sf(gene_cor_chisq, 1)
    data = {'INDEX': range(1, len(gene_cor) + 1),
            'H2': heri, 'H2_SE': heri_se,
            'H2_CHISQ': heri_chisq, 'H2_P': heri_pv,
            'GC': gene_cor, 'GC_SE': gene_cor_se,
            'GC_CHISQ': gene_cor_chisq, 'GC_P': gene_cor_pv}
    output = pd.DataFrame(data)
    
    return output


def print_results_two(heri_gc, output, overlap):
    msg = 'Heritability of the image\n'
    msg += '-------------------------\n'
    msg += (f"Mean h^2: {round(np.nanmean(output['H2']), 4)} "
            f"({round(np.nanmean(output['H2_SE']), 4)})\n")
    msg += f"Median h^2: {round(np.nanmedian(output['H2']), 4)}\n"
    msg += f"Max h^2: {round(np.nanmax(output['H2']), 4)}\n"
    msg += f"Min h^2: {round(np.nanmin(output['H2']), 4)}\n"
    msg += '\n'

    chisq_y2_heri = (heri_gc.y2_heri[0] / heri_gc.y2_heri_se[0]) ** 2
    pv_y2_heri = chi2.sf(chisq_y2_heri, 1)
    msg += 'Heritability of the non-imaging trait\n'
    msg += '-------------------------------------\n'
    msg += (f"Total observed scale h^2: {round(heri_gc.y2_heri[0], 4)} "
            f"({round(heri_gc.y2_heri_se[0], 4)})\n")
    msg += f"Chi^2: {round(chisq_y2_heri, 4)}\n"
    msg += f"P: {round(pv_y2_heri, 4)}\n"
    msg += '\n'

    if overlap:
        msg += 'Genetic correlation (with sample overlap)\n'
    else:
        msg += 'Genetic correlation (without sample overlap)\n'
    msg += '--------------------------------------------\n'
    msg += (f"Mean genetic correlation: {round(np.nanmean(output['GC']), 4)} "
            f"({round(np.nanmean(output['GC_SE']), 4)})\n")
    msg += f"Median genetic correlation: {round(np.nanmedian(output['GC']), 4)}\n"
    msg += f"Max genetic correlation: {round(np.nanmax(output['GC']), 4)}\n"
    msg += f"Min genetic correlation: {round(np.nanmin(output['GC']), 4)}\n"

    return msg


def print_results_heri(heri_output):
    msg = 'Heritability of the image\n'
    msg += '-------------------------\n'
    msg += (f"Mean heritability: {round(np.nanmean(heri_output['H2']), 4)} "
            f"({round(np.nanmean(heri_output['SE']), 4)})\n")
    msg += f"Median heritability: {round(np.nanmedian(heri_output['H2']), 4)}\n"
    msg += f"Max heritability: {round(np.nanmax(heri_output['H2']), 4)}\n"
    msg += f"Min heritability: {round(np.nanmin(heri_output['H2']), 4)}\n"

    return msg


def print_results_gc(mean_gene_cor, min_gene_cor, mean_gene_cor_se):
    msg = '\n'
    msg += 'Genetic correlation of the image\n'
    msg += '--------------------------------\n'
    msg += (f"Mean genetic correlation: {round(mean_gene_cor, 4)} "
            f"({round(mean_gene_cor_se, 4)})\n")
    # msg += f"Median genetic correlation: {round(np.nanmedian(gene_cor), 4)}\n"
    msg += f"Min genetic correlation: {round(min_gene_cor, 4)}\n"

    return msg


def run(args, log):
    check_input(args, log)
    
    # read LD matrices
    ld = LDmatrix(args.ld)
    log.info(f"Read LD matrix from {args.ld}")
    ld_inv = LDmatrix(args.ld_inv)
    log.info(f"Read LD inverse matrix from {args.ld_inv}")

    if ld.ldinfo.shape[0] != ld_inv.ldinfo.shape[0]:
        raise ValueError(('the LD matrix and LD inverse matrix have different number of SNPs. '
                          'It is highly likely that the files were misspecified or modified'))
    if not np.equal(ld.ldinfo[['A1', 'A2']].values, ld_inv.ldinfo[['A1', 'A2']].values).all():
        raise ValueError('LD matrix and LD inverse matrix have different alleles for some SNPs')
    log.info(f'{ld.ldinfo.shape[0]} SNPs read from LD matrix (and its inverse).')

    # read bases and inner_ldr
    bases = np.load(args.bases)
    log.info(f'{bases.shape[1]} bases read from {args.bases}')
    inner_ldr = np.load(args.inner_ldr)
    log.info(f'Read inner product of LDRs from {args.inner_ldr}')

    # read LDR gwas
    try:
        ldr_gwas = sumstats.read_sumstats(args.ldr_sumstats)
        log.info(f'{ldr_gwas.n_snps} SNPs read from LDR summary statistics {args.ldr_sumstats}')

        # keep selected LDRs
        if args.n_ldrs is not None:
            bases, inner_ldr, ldr_gwas = keep_ldrs(args.n_ldrs, bases, inner_ldr, ldr_gwas)
            log.info(f'Keep the top {args.n_ldrs} LDRs.')
        
        # check numbers of LDRs are the same
        if bases.shape[1] != inner_ldr.shape[0] or bases.shape[1] != ldr_gwas.n_gwas:
            raise ValueError(('inconsistent dimension for bases, inner product of LDRs, '
                            'and LDR summary statistics. '
                            'Try to use --n-ldrs'))

        # read y2 gwas
        if args.y2_sumstats:
            y2_gwas = sumstats.read_sumstats(args.y2_sumstats)
            log.info(f'{y2_gwas.n_snps} SNPs read from non-imaging summary statistics {args.y2_sumstats}')
        else:
            y2_gwas = None

        # extract SNPs
        if args.extract is not None:
            keep_snps = ds.read_extract(args.extract)
            log.info(f"{len(keep_snps)} SNPs in --extract.")
        else:
            keep_snps = None

        # get common snps from gwas, LD matrices, and keep_snps
        common_snps = get_common_snps(ldr_gwas, ld, ld_inv, y2_gwas, keep_snps) # slow
        log.info((f"{len(common_snps)} SNPs are common in these files with identical alleles. "
                "Extracting them from each file ..."))

        # extract common snps in LD matrix
        ld.extract(common_snps)
        ld_inv.extract(common_snps)

        # extract common snps in summary statistics and do alignment
        ldr_gwas.extract_snps(ld.ldinfo['SNP'])  # extract snp id
        ldr_gwas.align_alleles(ld.ldinfo) # get +/-

        if args.y2_sumstats:
            y2_gwas.extract_snps(ld.ldinfo['SNP'])
            y2_gwas.align_alleles(ld.ldinfo)
        log.info(f"Aligned genetic effects of summary statistics to the same allele.\n")

        log.info('Computing heritability and/or genetic correlation ...')
        if not args.y2_sumstats:
            heri_gc = OneSample(ldr_gwas, ld, ld_inv, bases, inner_ldr)
            heri_output = format_heri(heri_gc.heri, heri_gc.heri_se, log)
            msg = print_results_heri(heri_output)
            log.info(f'{msg}')
            heri_output.to_csv(f"{args.out}_heri.txt", 
                               sep='\t', index=None,
                               float_format='%.5e', na_rep='NA')
            log.info(f'Save the heritability results to {args.out}_heri.txt')

            if not args.heri_only:
                mean_gene_cor, min_gene_cor, mean_gene_cor_se = heri_gc.get_gene_cor_se(args.out)
                msg = print_results_gc(mean_gene_cor, min_gene_cor, mean_gene_cor_se)
                log.info(f'{msg}')
                log.info(f'Save the genetic correlation results to {args.out}_gc.h5')
        else:
            heri_gc = TwoSample(ldr_gwas, ld, ld_inv, bases, inner_ldr, y2_gwas, args.overlap)
            gene_cor_y2_output = format_gene_cor_y2(heri_gc.heri, heri_gc.heri_se,
                                                    heri_gc.gene_cor_y2,
                                                    heri_gc.gene_cor_y2_se, log)
            msg = print_results_two(heri_gc, gene_cor_y2_output, args.overlap)
            log.info(f'{msg}')
            gene_cor_y2_output.to_csv(f"{args.out}_gc.txt", sep='\t',
                                      index=None, float_format='%.5e',
                                      na_rep='NA')
            log.info(f'Save the genetic correlation results to {args.out}_gc.txt')
    finally:
        ldr_gwas.close()
        if args.y2_sumstats:
            y2_gwas.close()