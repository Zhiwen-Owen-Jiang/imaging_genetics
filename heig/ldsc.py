import numpy as np


class LDSC:
    """
    cross-trait LDSC for estimating the intercept when there is sample overlap
    
    """
    def __init__(self, ldr_sumstats_reader, y2_sumstats_reader, 
                 ldscore, ldr_heri, y2_heri, n1, n2, 
                 ld_rank, block_ranges, merged_blocks):
        n_blocks = len(merged_blocks)
        r = len(ldr_heri)
        n = np.sqrt(n1 * n2)
        self.lobo_ldsc = np.zeros((n_blocks, r))
        self.total_ldsc = np.zeros(r)

        y2_z_ldsc = next(y2_sumstats_reader)
        i = 0
        for ldr_z_batch in ldr_sumstats_reader:
            for j in range(ldr_z_batch.shape[1]):
                self.total_ldsc[i], self.lobo_ldsc[:, i] = self.ldsc(ldr_z_batch[:, j], y2_z_ldsc, n, n1, n2, 
                                                                     ldr_heri[i], y2_heri, ld_rank, 
                                                                     ldscore, block_ranges, merged_blocks)
                i += 1

    def ldsc(self, gwas1, gwas2, n, n1, n2, h1, h2, ld_rank, ldscore, block_ranges, merged_blocks):
        """
        Main LDSC estimator

        Parameters:
        ------------
        gwas1: a vector of Z score
        gwas2: a vector of Z score
        n: a vector of sqrt(n1 * n2)
        n1: a vector of sample size of gwas1
        n2: a vector of sample size of gwas2
        h1: heritability estimate of gwas1
        h2: heritability estimate of gwas2
        ldscore: a vector of ld score
        block_ranges: a list of block ranges 

        gwas1, gwas2 and ldscore should be aligned

        Returns:
        ---------
        coef_total[0]: intercept estimate using all data (1, )
        lobo_ldsc: left-one-block-out intercept estimates (n_blocks, )
        
        """
        y = gwas1 * gwas2
        init_h1, init_h2, init_gc, w_ldscore = self._process_input(y, h1, h2, n, n1, n2, ld_rank, ldscore)
        weights_part1 = (init_h1 * w_ldscore + 1) * (init_h2 * w_ldscore + 1)  # don't update
        weights_part2 = (init_gc * w_ldscore) ** 2  # need to update
        weights = 1 / (weights_part1 + weights_part2)
        weights *= 1 / w_ldscore
        X = np.stack((np.ones(len(ldscore)), ldscore * n / np.mean(n)), axis=1)

        # compute total WLS (update twice)
        for _ in range(2):
            xwx_total, xwy_total = self._wls(y, X, weights)
            coef_total = np.dot(np.linalg.inv(xwx_total), xwy_total)
            weights = self._update_weights(coef_total, n, ld_rank, w_ldscore, weights_part1)

        # compute left-one-block-out WLS
        lobo_ldsc = []
        for block in merged_blocks:
            begin, end = self._block_range(block, block_ranges)
            xwx_i, xwy_i = self._wls(
                y[begin: end], X[begin: end], weights[begin: end])
            coef = np.dot(np.linalg.inv(xwx_total - xwx_i), xwy_total - xwy_i)
            lobo_ldsc.append(coef[0])
        lobo_ldsc = np.array(lobo_ldsc)

        return coef_total[0], lobo_ldsc

    def _block_range(self, block, block_ranges):
        return block_ranges[block[0]][0], block_ranges[block[-1]][1]

    def _process_input(self, y, h1, h2, n, n1, n2, ld_rank, ldscore):
        init_gc = ld_rank * np.mean(y) / np.mean(ldscore * n)
        init_gc = min(init_gc, 1)
        init_gc = max(init_gc, -1)
        init_gc *= n / ld_rank
        h1 = min(h1, 1)
        h1 = max(h1, 0)
        h2 = min(h2, 1)
        h2 = max(h2, 0)
        init_h1 = h1 * n1 / ld_rank
        init_h2 = h2 * n2 / ld_rank
        ldscore[ldscore < 1] = 1

        return init_h1, init_h2, init_gc, ldscore

    def _wls(self, y, X, weights):
        X_w = X * weights.reshape(-1, 1)
        xwx = np.dot(X_w.T, X)
        xwy = np.dot(X_w.T, y)

        return xwx, xwy

    def _update_weights(self, coef, n, ld_rank, w_ldscore, weights_part1):
        rho_g = coef[1] * ld_rank / np.mean(n)
        rho_g = min(rho_g, 1)
        rho_g = max(rho_g, -1)
        weights_part2 = (rho_g * n / ld_rank * w_ldscore + coef[0]) ** 2
        weights = 1 / (weights_part1 + weights_part2)
        weights *= 1 / w_ldscore

        return weights

    def _remove_snps_chisq80(self, gwas):
        """
        gwas is a vector of Z score

        """
        gwas[gwas > np.sqrt(80)] = 0

        return np.array(gwas)
