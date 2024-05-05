import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

from heig import sumstats
import heig.input.dataset as ds
from heig.ldmatrix import LDmatrix
from heig.ldsc import LDSC


def align_alleles(ref, gwas):
    """
    Aligning the gwas2 with the current gwas such that 
    the Z scores are measured on the same allele.
    This function assumes that current gwas and gwas2 have
    identical SNPs, which happens after calling self._prune_snps()

    Parameters:
    ------------
    gwas1: a GWAS instance
    gwas2: a GWAS instance

    Returns:
    ---------
    gwas2: a GWAS instance with align alleles

    """
    if not (np.array(ref['SNP']) == np.array(gwas.snpinfo['SNP'])).all():
        raise ValueError("The GWAS and the reference have different SNPs.")

    aligned_z = [-gwas.z[i] if a11 != a12 else gwas.z[i]
                 for i, (a11, a12) in enumerate(zip(ref['A1'], gwas.snpinfo['A1']))]

    gwas.z = np.array(aligned_z)
    gwas.snpinfo['A1'] = ref['A1'].values
    gwas.snpinfo['A2'] = ref['A2'].values

    return gwas


def check_input(args, log):
    """
    Checking that all inputs are correct

    """
    # required arguments
    if args.ldr_sumstats is None:
        raise ValueError('--ldr-sumstats is required.')
    if args.bases is None:
        raise ValueError('--bases is required.')
    if args.inner_ldr is None:
        raise ValueError('--inner-ldr is required.')
    if args.ld_inv is None:
        raise ValueError('--ld-inv is required.')
    if args.ld is None:
        raise ValueError('--ld is required.')

    if not os.path.exists(f"{args.ldr_sumstats}.snpinfo"):
        raise FileNotFoundError(f"{args.ldr_sumstats}.snpinfo does not exist.")
    if not os.path.exists(f"{args.ldr_sumstats}.sumstats"):
        raise FileNotFoundError(
            f"{args.ldr_sumstats}.sumstats does not exist.")
    if not os.path.exists(args.bases):
        raise FileNotFoundError(f"{args.bases} does not exist.")
    if not os.path.exists(args.inner_ldr):
        raise FileNotFoundError(f"{args.inner_ldr} does not exist.")
    if args.overlap and not args.y2_sumstats:
        log.info('WARNING: ignore --overlap as --y2-sumstats is not specified.')
    if args.y2_sumstats is not None:
        if not os.path.exists(f"{args.y2_sumstats}.snpinfo"):
            raise FileNotFoundError(
                f"{args.y2_sumstats}.snpinfo does not exist.")
        if not os.path.exists(f"{args.y2_sumstats}.sumstats"):
            raise FileNotFoundError(
                f"{args.y2_sumstats}.sumstats does not exist.")


def get_common_snps(*snp_list):
    """
    Extracting common snps from multiple snp lists

    Parameters:
    ------------
    snp_list: multiple snp lists

    Returns:
    ---------
    common_snps: a list of common snp list

    """
    n_snp_list = len(snp_list)
    if n_snp_list == 0:
        raise ValueError('No snp list provided.')

    common_snps = None
    for i in range(len(snp_list)):
        if hasattr(snp_list[i], 'ldinfo'):
            snp = snp_list[i].ldinfo[['SNP', 'A1', 'A2']]
        elif hasattr(snp_list[i], 'snpinfo'):
            snp = snp_list[i].snpinfo[['SNP', 'A1', 'A2']]
        elif hasattr(snp_list[i], 'SNP'):
            snp = snp_list[i]['SNP']
        if not isinstance(common_snps, pd.DataFrame):
            common_snps = snp.copy()
        else:
            common_snps = common_snps.merge(snp, on='SNP')

    if common_snps is None:
        raise ValueError(
            'All the input snp lists are None or do not have a SNP column.')

    common_snps.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
    matched_alleles_set = common_snps[[col for col in common_snps.columns
                                       if col.startswith('A')]].apply(lambda x: len(set(x)) == 2, axis=1)
    common_snps = common_snps.loc[matched_alleles_set]

    return common_snps['SNP']


def read_process_data(args, log):
    """
    Reading and preprocessing gwas data

    """
    # read LD matrices
    log.info(f"Read LD matrix from {args.ld}")
    ld = LDmatrix(args.ld)
    log.info(f"Read LD inverse matrix from {args.ld_inv}")
    ld_inv = LDmatrix(args.ld_inv)
    if ld.ldinfo.shape[0] != ld_inv.ldinfo.shape[0]:
        raise ValueError(('The LD matrix and LD inverse matrix have different number of SNPs. ',
                          'It is highly likely that the files were misspecified or modified.'))
    if (not (np.array(ld.ldinfo['A1']) == np.array(ld_inv.ldinfo['A1'])).all()
            or not (np.array(ld.ldinfo['A2']) == np.array(ld_inv.ldinfo['A2'])).all()):
        raise ValueError(
            'LD matrix and LD inverse matrix have different alleles for some SNPs.')
    log.info(
        f'{ld.ldinfo.shape[0]} SNPs read from LD matrix (and its inverse).')

    # read bases and inner_ldr
    bases = np.load(args.bases)
    log.info(f'{bases.shape[1]} bases read from {args.bases}')
    if bases.shape[1] < args.n_ldrs:
        raise ValueError('The number of bases is less than the number of LDR.')
    bases = bases[:, :args.n_ldrs]

    log.info(f'Read inner product of LDRs from {args.inner_ldr}')
    inner_ldr = np.load(args.inner_ldr)
    if inner_ldr.shape[0] < args.n_ldrs or inner_ldr.shape[1] < args.n_ldrs:
        raise ValueError(
            'The dimension of inner product of LDR is less than the number of LDR.')
    inner_ldr = inner_ldr[:args.n_ldrs, :args.n_ldrs]
    log.info(f'Keep the top {args.n_ldrs} LDRs.\n')

    # read LDR gwas
    ldr_gwas = sumstats.read_sumstats(args.ldr_sumstats)
    ldr_gwas.get_zscore()
    log.info(
        f'{ldr_gwas.snpinfo.shape[0]} SNPs read from LDR summary statistics {args.ldr_sumstats}')

    # keep selected LDRs
    if args.n_ldrs:
        if args.n_ldrs > ldr_gwas.z.shape[1]:
            raise ValueError(
                '--n-ldrs is greater than LDRs in summary statistics.')
        else:
            ldr_gwas.z = ldr_gwas.z[:, :args.n_ldrs]

    # read y2 gwas
    if args.y2_sumstats:
        y2_gwas = sumstats.read_sumstats(args.y2_sumstats)
        log.info((f'{y2_gwas.snpinfo.shape[0]} SNPs read from non-imaging '
                  f'summary statistics {args.y2_sumstats}'))
    else:
        y2_gwas = None

     # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} SNPs are common in --extract.")
    else:
        keep_snps = None

    # get common snps from gwas, LD matrices, and keep_snps
    common_snps = get_common_snps(ldr_gwas, ld, ld_inv,
                                  y2_gwas, keep_snps)
    log.info((f"{len(common_snps)} SNPs are common in these files with identical alleles. "
              "Extracting them from each file ..."))

    # extract common snps in LD matrix
    ld.extract(common_snps)
    ld_inv.extract(common_snps)

    # extract common snps in summary statistics and do alignment
    ldr_gwas.extract_snps(ld.ldinfo['SNP'])  # TODO: speed up
    ldr_gwas = align_alleles(ld.ldinfo, ldr_gwas)

    if args.y2_sumstats:
        y2_gwas.extract_snps(ld.ldinfo['SNP'])
        y2_gwas = align_alleles(ld.ldinfo, y2_gwas)
    log.info(f"Aligned genetic effects of summary statistics to the same allele.\n")

    return ldr_gwas, y2_gwas, bases, inner_ldr, ld, ld_inv


class Estimation:
    def __init__(self, z_mat, n, ld, ld_inv, bases, inner_ldr):
        """
        z_mat and y2_z have been normalized 

        """
        self.z_mat = z_mat
        self.n = np.array(n)
        self.nbar = np.mean(n)
        self.ld = ld
        self.ld_inv = ld_inv
        self.r = z_mat.shape[1]
        self.bases = bases
        self.inner_ldr = inner_ldr
        self.block_ranges = ld.block_ranges

        self.sigmaX_cov = np.dot(np.dot(bases, inner_ldr), bases.T) / self.nbar
        self.sigmaX_var = np.diag(self.sigmaX_cov)

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

    def _block_wise_estimate(self):
        raise NotImplementedError


class OneSample(Estimation):
    def __init__(self, z_mat, n, ld, ld_inv, bases, inner_ldr, heri_only):
        super().__init__(z_mat, n, ld, ld_inv, bases, inner_ldr)

        self.ld_block_rank = np.zeros(len(self.block_ranges))
        self.ldr_block_gene_cov = np.zeros(
            (len(self.ld_block_rank), self.r, self.r))
        for i, ((begin, end), ld_block, ld_inv_block) in enumerate(zip(self.block_ranges,
                                                                       self.ld.data,
                                                                       self.ld_inv.data)):
            ld_block_rank, block_gene_cov = self._block_wise_estimate(
                begin, end, ld_block, ld_inv_block
            )
            self.ld_block_rank[i] = ld_block_rank
            self.ldr_block_gene_cov[i, :, :] = block_gene_cov
        self.ld_rank = np.sum(self.ld_block_rank)
        self.ldr_gene_cov = np.sum(self.ldr_block_gene_cov, axis=0)

        self.gene_cov = np.dot(
            np.dot(self.bases, self.ldr_gene_cov), self.bases.T)
        self.heri = np.diag(self.gene_cov) / self.sigmaX_var
        sigmaEta_cov = self.sigmaX_cov - self.gene_cov

        self.heri_se = self._get_heri_se(self.heri, self.ld_rank, self.nbar)
        if not heri_only:
            self.gene_cor = self._get_gene_cor()
            self.gene_cor_se = self._get_gene_cor_se(
                self.heri, self.gene_cor, self.gene_cov,
                sigmaEta_cov, self.ld_rank, self.nbar
            )
        else:
            self.gene_cor, self.gene_cor_se = None, None

    def _block_wise_estimate(self, begin, end, ld_block, ld_block_inv):
        """
        block_ld: eigenvectors * sqrt(eigenvalues)
        block_ld_inv: eigenvectors * sqrt(eigenvalues ** -1)

        """
        block_ld_ld_inv = np.dot(ld_block.T, ld_block_inv)
        ld_block_rank = np.sum(block_ld_ld_inv ** 2)

        z_mat_block_ld_inv = np.dot(self.z_mat[begin: end, :].T, ld_block_inv)
        block_gene_cov = (np.dot(z_mat_block_ld_inv, z_mat_block_ld_inv.T) -
                          ld_block_rank * self.inner_ldr / self.nbar ** 2)

        return ld_block_rank, block_gene_cov

    def _get_gene_cor(self):
        gene_var = np.diag(self.gene_cov)
        gene_cor = self.gene_cov / np.sqrt(np.outer(gene_var, gene_var))

        return gene_cor

    def _get_gene_cor_se(self, heri, gene_cor, gene_cov, sigmaEta_cov, d, n):
        """
        Estimating standard error of one-sample genetic correlation estimates 

        Parameters:
        ------------
        heri: an N by 1 vector of heritability estimates of images
        gene_cor: an N by N matrix of genetic correlation estimates of images
        gene_cov: an N by N matrix of genetic covariance estimates of images
        sigmaEta_cov: an N by N matrix of sigmaEta estimates (the residual variance of an image)
        d: number of SNPs, or Tr(R\Omega)  
        n: sample size

        Returns:
        ---------
        An N by N matrix of standard error estimates

        """
        heri[heri <= 0] = np.nan
        gene_cov[gene_cov == 0] = 0.01  # just to avoid division by zero error

        N = len(heri)
        inv_heri = np.repeat(1 / heri - 1, N).reshape(N, N)
        gene_cor2 = gene_cor ** 2
        part1 = 1 - gene_cor2
        part2 = sigmaEta_cov / gene_cov
        part3 = inv_heri + inv_heri.T

        res = ((4 / n + d / n**2) * part1**2 +
               (1 / n + d / n**2) * part1 * (part3 - 2 * gene_cor2 * part2) +
               d / n**2 / 2 * gene_cor2 * (inv_heri - part2)**2 +
               d / n**2 / 2 * gene_cor2 * (inv_heri.T - part2)**2 +
               d / n**2 * (gene_cor2**2 * part2**2 + inv_heri * inv_heri.T) -
               d / n**2 * gene_cor2 * part2 * part3)
        res[np.abs(res) < 10 ** -10] = 0

        return np.sqrt(res)


class TwoSample(Estimation):
    def __init__(self, z_mat, n, ld, ld_inv, bases, inner_ldr,
                 y2_z, n2, overlap=False):
        super().__init__(z_mat, n, ld, ld_inv, bases, inner_ldr)
        self.y2_z = y2_z
        self.n2 = np.array(n2)
        self.n2bar = np.mean(n2)

        y2_block_gene_cov = np.zeros(len(self.block_ranges))
        ldr_y2_block_gene_cov_part1 = np.zeros(
            (len(self.block_ranges), self.r))
        self.ld_block_rank = np.zeros(len(self.block_ranges))
        self.ldr_block_gene_cov = np.zeros(
            (len(self.ld_block_rank), self.r, self.r))
        for i, ((begin, end), ld_block, ld_inv_block) in enumerate(zip(self.block_ranges,
                                                                       self.ld.data,
                                                                       self.ld_inv.data)):
            (ld_block_rank,
             block_gene_var_y2,
             block_gene_cov,
             block_gene_cov_y2
             ) = self._block_wise_estimate(begin, end, ld_block, ld_inv_block)
            self.ld_block_rank[i] = ld_block_rank
            y2_block_gene_cov[i] = block_gene_var_y2
            ldr_y2_block_gene_cov_part1[i, :] = block_gene_cov_y2
            self.ldr_block_gene_cov[i, :, :] = block_gene_cov
        self.ld_rank = np.sum(self.ld_block_rank)
        self.ldr_gene_cov = np.sum(self.ldr_block_gene_cov, axis=0)

        # since the sum stats have been normalized
        self.y2_heri = np.atleast_1d(np.sum(y2_block_gene_cov))

        ldr_y2_gene_cov_part1 = np.sum(ldr_y2_block_gene_cov_part1, axis=0)

        self.heri = (np.sum(np.dot(self.bases, self.ldr_gene_cov) * self.bases, axis=1) /
                     self.sigmaX_var)
        self.heri_se = self._get_heri_se(self.heri, self.ld_rank, self.nbar)

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
            y2_lobo_heri = self._lobo_estimate(
                self.y2_heri, y2_block_gene_cov, merged_blocks)
            temp = np.matmul(np.swapaxes(ldr_lobo_gene_cov,
                             1, 2), bases.T).swapaxes(1, 2)
            temp = np.sum(temp * np.expand_dims(bases, 0), axis=2)
            image_lobo_heri = temp / self.sigmaX_var

            # compute left-one-block-out cross-trait LDSC intercept
            self.ldr_heri = np.diag(self.ldr_gene_cov) / \
                np.diag(self.inner_ldr) * self.nbar
            z_mat_raw = self.z_mat / \
                np.sqrt(np.diagonal(inner_ldr)) * self.n.reshape(-1, 1)
            y2_z_raw = self.y2_z * np.sqrt(self.n2).reshape(-1, 1)
            ldsc_intercept = LDSC(z_mat_raw, y2_z_raw, ldscore, self.ldr_heri, self.y2_heri,
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

        self.y2_heri_se = self._get_heri_se(
            self.y2_heri, self.ld_rank, self.n2bar)

    def _block_wise_estimate(self, begin, end, ld_block, ld_block_inv):
        """
        ld_block: eigenvectors * sqrt(eigenvalues)
        ld_block_inv: eigenvectors * sqrt(eigenvalues ** -1)

        """
        ld_block_ld_inv = np.dot(ld_block.T, ld_block_inv)
        ld_block_rank = np.sum(ld_block_ld_inv ** 2)

        z_mat_ld_block_inv = np.dot(self.z_mat[begin: end, :].T, ld_block_inv)
        y2_ld_block_inv = np.dot(self.y2_z[begin: end].T, ld_block_inv)

        block_gene_var_y2 = np.dot(
            y2_ld_block_inv, y2_ld_block_inv.T) - ld_block_rank / self.n2bar
        block_gene_cov_y2 = np.squeeze(
            np.dot(z_mat_ld_block_inv, y2_ld_block_inv.T))
        block_gene_cov = (np.dot(z_mat_ld_block_inv, z_mat_ld_block_inv.T) -
                          ld_block_rank * self.inner_ldr / self.nbar ** 2)

        return ld_block_rank, block_gene_var_y2, block_gene_cov, block_gene_cov_y2

    def _get_gene_cor_ldsc(self, part1, ld_rank, ldsc, heri1, heri2):
        ldr_gene_cov = (part1 - ld_rank.reshape(-1, 1) * ldsc *
                        np.sqrt(np.diagonal(self.inner_ldr)) / self.nbar / np.sqrt(self.n2bar))
        gene_cor = self._get_gene_cor_y2(ldr_gene_cov.T, heri1, heri2)

        return gene_cor

    def _get_gene_cor_y2(self, inner_part, heri1, heri2):
        bases_inner_part = np.dot(self.bases, inner_part).reshape(
            self.bases.shape[0], -1)
        gene_cov_y2 = bases_inner_part / \
            np.sqrt(self.sigmaX_var).reshape(-1, 1)
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
        gene_cor_y2sq = gene_cor_y2 ** 2
        gene_cor_y2sq1 = 1 - gene_cor_y2sq
        part1 = gene_cor_y2sq / (2 * heri ** 2) * d / n1 ** 2
        part2 = gene_cor_y2sq / (2 * heri_y2 ** 2) * d / n2 ** 2
        part3 = 1 / (heri * heri_y2) * d / (n1 * n2)
        part4 = gene_cor_y2sq1 / (heri * n1)
        part5 = gene_cor_y2sq1 / (heri_y2 * n2)
        var = part1 + part2 + part3 + part4 + part5

        return np.sqrt(var)


def format_heri(heri, heri_se, log):
    invalid_heri = (heri > 1) | (heri < 0) | (heri_se > 1) | (heri_se < 0)
    heri[invalid_heri] = np.nan
    heri_se[invalid_heri] = np.nan
    log.info('Removed out-of-bound results (if any)\n')

    chisq = (heri / heri_se) ** 2
    pv = chi2.sf(chisq, 1)
    data = {'INDEX': range(1, len(heri) + 1), 'H2': heri, 'SE': heri_se,
            'CHISQ': chisq, 'P': pv}
    output = pd.DataFrame(data)
    return output


def format_gene_cor(gene_cor, gene_cor_se):
    gene_cor_tril = gene_cor[np.tril_indices(gene_cor.shape[0], k=-1)]
    gene_cor_se_tril = gene_cor_se[np.tril_indices(gene_cor_se.shape[0], k=-1)]
    invalid_gene_cor = ((gene_cor_tril > 1) |
                        (gene_cor_tril < -1) |
                        (gene_cor_se_tril > 1) |
                        (gene_cor_se_tril < 0))
    gene_cor_tril[invalid_gene_cor] = np.nan
    gene_cor_se_tril[invalid_gene_cor] = np.nan

    return gene_cor_tril, gene_cor_se_tril


def format_gene_cor_y2(heri, heri_se, gene_cor, gene_cor_se, log):
    invalid_heri = (heri > 1) | (heri < 0) | (heri_se > 1) | (heri_se < 0)
    heri[invalid_heri] = np.nan
    heri_se[invalid_heri] = np.nan
    invalid_gene_cor = ((gene_cor > 1) |
                        (gene_cor < -1) |
                        (gene_cor_se > 1) |
                        (gene_cor_se < 0))
    gene_cor[invalid_gene_cor] = np.nan
    gene_cor_se[invalid_gene_cor] = np.nan
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
    msg += (f"Mean h^2: {round(np.nanmean(heri_output['H2']), 4)} "
            f"({round(np.nanmean(heri_output['SE']), 4)})\n")
    msg += f"Median h^2: {round(np.nanmedian(heri_output['H2']), 4)}\n"
    msg += f"Max h^2: {round(np.nanmax(heri_output['H2']), 4)}\n"
    msg += f"Min h^2: {round(np.nanmin(heri_output['H2']), 4)}\n"

    return msg


def print_results_gc(gene_cor, gene_cor_se):
    msg = '\n'
    msg += 'Genetic correlation of the image\n'
    msg += '--------------------------------\n'
    msg += (f"Mean genetic correlation: {round(np.nanmean(gene_cor), 4)} "
            f"({round(np.nanmean(gene_cor_se), 4)})\n")
    msg += f"Median genetic correlation: {round(np.nanmedian(gene_cor), 4)}\n"
    msg += f"Min genetic correlation: {round(np.nanmin(gene_cor), 4)}\n"

    return msg


def run(args, log):
    check_input(args, log)
    ldr_gwas, y2_gwas, bases, inner_ldr, ld, ld_inv = read_process_data(
        args, log)
    log.info('Computing heritability and/or genetic correlation ...')

    # normalize summary statistics of LDR
    z_mat = (ldr_gwas.z * np.sqrt(np.diagonal(inner_ldr)) /
             np.array(ldr_gwas.snpinfo['N']).reshape(-1, 1))

    if not args.y2_sumstats:
        heri_gc = OneSample(z_mat, ldr_gwas.snpinfo['N'], ld,
                            ld_inv, bases, inner_ldr,
                            args.heri_only)
        heri_output = format_heri(heri_gc.heri, heri_gc.heri_se, log)
        msg = print_results_heri(heri_output)
        log.info(f'{msg}')
        heri_output.to_csv(f"{args.out}_heri.txt",
                           sep='\t', index=None,
                           float_format='%.5e', na_rep='NA')
        log.info(f'Save the heritability results to {args.out}_heri.txt')

        if not args.heri_only:
            gene_cor_tril, gene_cor_se_tril = format_gene_cor(heri_gc.gene_cor,
                                                              heri_gc.gene_cor_se)
            msg = print_results_gc(heri_gc.gene_cor, heri_gc.gene_cor_se)
            log.info(f'{msg}')
            np.savez_compressed(
                f'{args.out}_gc', gc=gene_cor_tril, se=gene_cor_se_tril)
            log.info(
                f'Save the genetic correlation results to {args.out}_gc.npz')
    else:
        y2_z = y2_gwas.z / \
            np.sqrt(np.array(y2_gwas.snpinfo['N'])).reshape(-1, 1)
        heri_gc = TwoSample(z_mat, ldr_gwas.snpinfo['N'], ld, ld_inv, bases, inner_ldr,
                            y2_z, y2_gwas.snpinfo['N'], args.overlap)
        gene_cor_y2_output = format_gene_cor_y2(heri_gc.heri, heri_gc.heri_se,
                                                heri_gc.gene_cor_y2,
                                                heri_gc.gene_cor_y2_se, log)
        msg = print_results_two(heri_gc, gene_cor_y2_output, args.overlap)
        log.info(f'{msg}')
        gene_cor_y2_output.to_csv(f"{args.out}_gc.txt", sep='\t',
                                  index=None, float_format='%.5e',
                                  na_rep='NA')
        log.info(f'Save the genetic correlation results to {args.out}_gc.txt')
