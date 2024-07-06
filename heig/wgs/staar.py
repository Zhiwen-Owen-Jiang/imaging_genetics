import numpy as np
import pandas as pd
from scipy.stats import chi2, cauchy, beta
from heig.wgs.pvalue import saddle
from hail.linalg import BlockMatrix


"""
TODO:
1. try to compute quantiles for a mixture of chisq1 distribution
and use this quantile to quickly screen insignificant results

"""

class VariantSetTest:
    def __init__(self, bases, inner_ldr, resid_ldr, covar, var, block_size=2048):
        """
        bases: (N, r) np.array, functional bases
        inner_ldr: (r, r) np.array, inner product of projected LDRs
        resid_ldr: (n, r) np.array, LDR residuals
        covar: (n, p) np.array, the same as those used to do projection
        var: (N, ) np.array, voxel-level variance
        block_size: block size for BlockMatrix

        """
        self.bases = bases
        self.inner_ldr = inner_ldr
        self.covar = BlockMatrix.from_numpy(covar, block_size=block_size)
        self.var = var
        self.n, self.p = covar.shape
        self.resid_ldr = BlockMatrix.from_numpy(resid_ldr, block_size=block_size)
        self.N = bases.shape[0]
        self.block_size = block_size

        # null model
        self.inner_covar_inv = BlockMatrix.from_numpy(np.linalg.inv(np.dot(covar.T, covar)), block_size=block_size) # (p, p)
        self.covar_ldr = BlockMatrix.from_numpy(np.dot(covar.T, resid_ldr), block_size=block_size) # (p, r)

    def input_vset(self, vset, annotation_pred=None):
        """
        Inputing variant set and computing half scores and covariance matrix

        Parameters:
        ------------
        vset: (n, m) preprocessed MatrixTable with alleles flipped and variants filtered.
        It should include only variants of a specific category or in a small window, 
        not the entire gene. Otherwise, there might be OOM issue.
        annotation_pred: (q, m) np.array of functional annotation or None

        """
        self.maf = np.array(vset.maf.collect())
        self.is_rare = np.array(vset.is_rare.collect())
        vset = BlockMatrix.from_entry_expr(vset.flipped_n_alt_alleles, mean_impute=True, 
                                           block_size=self.block_size) # (m, n)
        vset_covar = vset @ self.covar  # Z'X, (m, p)
        inner_vset = vset @ vset.T  # Z'Z, (m, m)
        half_ldr_score = vset @ self.resid_ldr # Z'(I-M)\Xi, (m, r)
        cov_mat = inner_vset - vset_covar @ self.inner_covar_inv @ vset_covar.T  # Z'(I-M)Z, (m, m)

        self.half_ldr_score = half_ldr_score.to_numpy() #  (m, r)
        self.half_score = np.dot(self.half_ldr_score, self.bases.T) # (m, N)
        self.cov_mat = cov_mat.to_numpy()
        self.weights = self._get_weights(annotation_pred)
        self.n_variants = self.half_ldr_score.shape[0]

    def _get_weights(self, annot=None):
        """
        vertically stack weights, i.e., each row is a (m, ) vector

        Parameters:
        ------------
        annot: (q, m) array, l is #functional weights, m is #variants

        Returns:
        ---------
        weights_dict: a dict of weights
        
        """
        w1 = beta.pdf(self.maf, 1, 25)
        w2 = beta.pdf(self.maf, 1, 1)
        w3 = beta.pdf(self.maf, 0.5, 0.5)
        weights_dict = dict()

        if annot is None:
            weights_dict['skat(1,25)'] = w1
            weights_dict['skat(1,1)'] = w2
            weights_dict['burden(1,25)'] = w1
            weights_dict['burden(1,1)'] = w2
            weights_dict['acatv(1,25)'] = w1 / w3 ** 2
            weights_dict['acatv(1,25)'] = w2 / w3 ** 2
        else:
            annot_rank = 1 - 10 ** (-annot/10)
            annot_rank = annot_rank.T
            weights_dict['skat(1,25)'] = self._combine_weights(w1, np.sqrt(annot_rank))
            weights_dict['skat(1,1)'] = self._combine_weights(w2, np.sqrt(annot_rank))
            weights_dict['burden(1,25)'] = self._combine_weights(w1, annot_rank)
            weights_dict['burden(1,1)'] = self._combine_weights(w2, annot_rank)
            weights_dict['acatv(1,25)'] = self._combine_weights((w1 / w3) ** 2, annot_rank)
            weights_dict['acatv(1,1)'] = self._combine_weights((w2 / w3) ** 2, annot_rank)

        return weights_dict

    def _combine_weights(self, w, w_annot):
        w_annot = w_annot * w
        weights = np.vstack([w, w_annot])
        return weights

    def _skat_test(self, weights):
        """
        Computing SKAT pvalues for one weight and all voxels.

        For a single trait Y, the score test statistic of SKAT:
        Y'(I-M)ZWWZ'(I-M)Y/\sigma^2.
        W is a m by m diagonal matrix, Z is a n by m matrix,
        M = X(X'X)^{-1}X'.
        Under the null, it follows a mixture of chisq(1) distribution,
        where the weights are eigenvalues of WZ'(I-M)ZW.
        All voxels share the same eigenvalues.
        Use Saddle-point approximation to compute pvalues.

        Parameters:
        ------------
        weights: (m, ) array

        Returns:
        ---------
        pvalues: (N, ) array

        """
        weighted_half_ldr_score = weights.reshape(-1, 1) * self.half_ldr_score  # (m, r)
        ldr_score = np.dot(weighted_half_ldr_score.T,
                           weighted_half_ldr_score)  # (r, r)
        score_stat = np.sum(np.dot(self.bases, ldr_score) * self.bases, axis=1) / self.var # (N, )

        wcov_mat = weights.reshape(-1, 1) * self.cov_mat * weights  # (m, m)
        egvalues, _ = np.linalg.eigh(wcov_mat)
        # (m, ) all voxels share the same eigenvalues
        egvalues = np.flip(egvalues)
        egvalues[egvalues < 10**-8] = 0

        pvalues = saddle(score_stat, egvalues, wcov_mat)
        return pvalues

    def _burden_test(self, weights):
        """
        Computing Burden pvalues for one weight and all voxels.

        For a single trait Y, the burden test statistic
        (w'Z'(I-M)Y)^2 / (\hat{\sigma}^2 w'Z'(I-M)Zw)
        where w is a m by 1 vector, Z is a n by m matrix,
        M = X(X'X)^{-1}X'.
        Under the null, it follows a chisq(1) distribution.

        Parameters:
        ------------
        weights: (m, ) array

        Returns:
        ---------
        pvalues: (N, ) array

        """
        burden_score_num = np.dot(weights, self.half_score) ** 2  # (N, )
        burden_score_denom = self.var * np.dot(np.dot(weights, self.cov_mat), weights)
        burden_score = burden_score_num / burden_score_denom  # (N, )
        pvalues = chi2.sf(burden_score, 1)
        return pvalues

    def _acatv_test(self, weights):
        """
        Computing ACATV pvalues for one weight and all voxels.

        First split the variant set to very rare (MAC <= 10) and common sets.
        For very rare sets, do Burden test; for common set, do individual test,
        then combine the pvalues by cauchy combination.
        
        Parameters:
        ------------
        weights: (m, ) array

        Returns:
        ---------
        pvalues: (N, ) array
        
        """
        all_weights = weights ** 2 * self.maf * (1 - self.maf)

        ## score test for individual common variants
        if (~self.is_rare).any():
            denom = np.diag(self.cov_mat[~self.is_rare][:, ~self.is_rare]).reshape(-1, 1) * self.var.reshape(1, -1)  # (m, N)
            common_variant_pv = chi2.sf((self.half_score[~self.is_rare] ** 2 / denom[~self.is_rare]), 1)  # (m1, N)
            common_weights = all_weights[~self.is_rare] # (m1, )
        else:
            common_variant_pv = None
            common_weights = None

        ## Burden test for rare variants
        if (self.is_rare).any():
            rare_burden_score_num = np.dot(weights[self.is_rare], self.half_score[self.is_rare]) ** 2  # (N, )
            rare_burden_score_denom = self.var * \
                np.dot(np.dot(weights[self.is_rare], self.cov_mat[self.is_rare]
                    [:, self.is_rare]), weights[self.is_rare])  # (N, )
            rare_burden_score = rare_burden_score_num / rare_burden_score_denom
            rare_burden_pv = chi2.sf(rare_burden_score, 1).reshape(1, -1) # (1, N)
            rare_weights = np.mean(all_weights[self.is_rare]) # (1, )
        else:
            rare_burden_pv = None
            rare_weights = None

        if common_variant_pv is not None and rare_burden_pv is not None:
            pvalues = cauchy_combination(np.vstack([common_variant_pv, rare_burden_pv]),
                                         np.vstack([common_weights, rare_weights]))
        elif common_variant_pv is not None:
            pvalues = cauchy_combination(common_variant_pv, common_weights)
        else:
            pvalues = rare_burden_pv

        return pvalues

    def do_inference(self, anno_name):
        """
        Doing inference for the variant set using multiple weights and methods.
        Using cauchy combination to get final pvalues.

        Returns:
        ---------
        results: a dict of results, each value is a np.array of pvalues (q+1, N)

        """
        n_weights = self.weights['skat(1,25)'].shape[0]
        skat_1_25_pvalues = np.zeros((n_weights, self.N))
        skat_1_1_pvalues = np.zeros((n_weights, self.N))
        burden_1_25_pvalues = np.zeros((n_weights, self.N))
        burden_1_1_pvalues = np.zeros((n_weights, self.N))
        acatv_1_25_pvalues = np.zeros((n_weights, self.N))
        acatv_1_1_pvalues = np.zeros((n_weights, self.N))

        for i in range(n_weights):
            skat_1_25_pvalues[i] = self._skat_test(self.weights['skat(1,25)'][i])
            skat_1_1_pvalues[i] = self._skat_test(self.weights['skat(1,1)'][i])
            burden_1_25_pvalues[i] = self._burden_test(self.weights['burden(1,25)'][i])
            burden_1_1_pvalues[i] = self._burden_test(self.weights['burden(1,1)'][i])
            acatv_1_25_pvalues[i] = self._acatv_test(self.weights['acatv(1,25)'][i])
            acatv_1_1_pvalues[i] = self._acatv_test(self.weights['acatv(1,1)'][i])

        all_pvalues = np.vstack([skat_1_25_pvalues, skat_1_1_pvalues, 
                                 burden_1_25_pvalues, burden_1_1_pvalues, 
                                 acatv_1_25_pvalues, acatv_1_1_pvalues])
        results_STAAR_O = pd.DataFrame(cauchy_combination(all_pvalues), columns=['STAAR-O'])
        results_ACAT_O = cauchy_combination(np.vstack([skat_1_25_pvalues[0], skat_1_1_pvalues[0],
                                                       burden_1_25_pvalues[0], burden_1_1_pvalues[0],
                                                       acatv_1_25_pvalues[0], acatv_1_1_pvalues[0]]))
        results_ACAT_O = pd.DataFrame(results_ACAT_O, columns=['ACAT-O'])
        all_results = [results_STAAR_O, results_ACAT_O]

        for pvalues, test_method in ((skat_1_25_pvalues, 'SKAT(1,25)'), (skat_1_1_pvalues, 'SKAT(1,1)'), 
                                     (burden_1_25_pvalues, 'Burden(1,25)'), (burden_1_1_pvalues, 'Burden(1,1)'),
                                     (acatv_1_25_pvalues, 'AVAT-V(1,25)'), (acatv_1_1_pvalues, 'AVAT-V(1,1)')):
            if n_weights > 1:
                comb_pvalues = cauchy_combination(pvalues)
            else:
                comb_pvalues = None
            all_pvalues = format_results(pvalues, comb_pvalues, test_method, anno_name)
            all_results.append(all_pvalues)
        all_results_df = pd.concat(*all_results, axis=1)

        return all_results_df


def cauchy_combination(pvalues, weights=None, axis=0):
    """
    Cauchy combination for an array of pvalues and weights.

    Parameters:
    ------------
    pvalues: (m1, N) array
    weights: (m1, ) array
    axis: which axis to take average

    Returns:
    ---------
    cct_pvalues: (N, ) array

    """
    if weights is None:
        weights = np.ones(pvalues.shape)
    normed_weights = weights / np.sum(weights, axis=axis)
    if len(normed_weights.shape) == 1:
        normed_weights = normed_weights.reshape(-1, 1)
    stats = np.mean(np.tan((0.5 - pvalues) * np.pi) * normed_weights, 
                    axis=axis)
    cct_pvalues = cauchy.sf(stats, loc=0, scale=1)

    return cct_pvalues


def remove_relatedness(ldrs, chr_preds, chr):
    mask = np.ones(chr_preds.shape[2], dtype=bool)
    mask[chr-1] = False
    loco_preds = np.sum(chr_preds[:, :, mask], axis=2)

    return ldrs - loco_preds


def format_results(indiv_pvalues, comb_pvalues, method_name, indiv_annot_name):
    """
    Formating test results. Individual p-values go first followed by combined p-values.

    Parameters:
    ------------
    indiv_pvalues: (N, 1) array
    comb_pvalues: (N, ) array
    indiv_annot_name: name of each functional annotation
    method_name: test method with beta distribution parameters, such as SKAT(1,25)

    Returns:
    ---------
    cct_pvalues: (N, ) array
    
    """
    if comb_pvalues is not None:
        col_names = [method_name]
        for annot in indiv_annot_name:
            col_names.append(f"{method_name}-{annot}")
        col_names.append(f"STAAR-{method_name}")
        res = pd.DataFrame(np.column_stack([indiv_pvalues, comb_pvalues]), columns=col_names)
    else:
        res = pd.DataFrame(indiv_pvalues, columns=method_name)

    return res