import os
import numpy as np
import pandas as pd
from scipy.stats import chi2, cauchy



class VariantSetTest:
    def __init__(self, bases, inner_ldr, ldrs, covar):
        """
        bases: N by r matrix, functional bases
        inner_ldr: r by r matrix, inner product of projected LDRs
        ldrs: n by r matrix, unprojected LDRs
        covar: n by p matrix, the same as those used to do projection
        
        """
        self.bases = bases
        self.inner_ldr = inner_ldr
        self.covar = covar
        self.n, self.p = covar.shape
        self.ldrs = ldrs

        # null model
        self.inner_covar_inv = np.linalg.inv(np.dot(covar.T, covar)) # p by p
        self.covar_ldrs = np.dot(covar.T, ldrs) # p by r
        self.var = np.sum(np.dot(self.bases, self.inner_ldr) * self.bases.T, axis=1) / (self.n - self.p) # N by 1


    def input_vset(self, vset, weights):
        """
        vset: n by m matrix, variant set
        weights: m by 1 array, weight for each variant

        """

        self.mac = np.sum(vset > 0, axis=0)
        self.maf = np.mean(vset, axis=0) / 2
        self.vset = vset
        self.weights = weights
        self.vset_covar = np.dot(self.vset.T, self.covar) # Z'X, m by p
        self.vset_ldrs = np.dot(self.vset.T, self.ldrs) # Z'\Xi, m by r
        self.inner_vset = np.dot(self.vset.T, self.vset) # Z'Z, m by m
        self.half_ldr_score = self.vset_ldrs - np.dot(np.dot(self.vset_covar, self.inner_covar_inv), 
                                             self.covar_ldrs) # Z'(I-M)\Xi, m by r
        self.adj_mat = self.inner_vset - np.dot(np.dot(self.vset_covar, self.inner_covar_inv), 
                                             self.vset_covar.T) # Z'(I-M)Z, m by m
    

    def _skat_test(self):
        """
        For a single trait Y, the score test statistic of SKAT:
        Y'(I-M)ZWWZ'(I-M)Y / \hat{\sigma}^4.
        W is a m by m diagonal matrix, Z is a n by m matrix,
        M = X(X'X)^{-1}X'.
        Under the null, it follows a mixture of chisq(1) distribution,
        where the weights are eigen values of WZ'(I-M)ZW.

        Use Saddle-point approximation to compute pvalues.
        TODO: standardize the weights? Then computing pvalues will be much faster.

        """
        weighted_half_ldr_score = self.weights.reshape(-1, 1) * self.half_ldr_score
        ldr_score = np.dot(weighted_half_ldr_score.T, weighted_half_ldr_score) # r by r
        score_stat = np.sum(np.dot(self.bases, ldr_score) * self.bases.T, axis=1) / self.var ** 2 # N by 1

        w_mat = self.weights.reshape(-1, 1) * self.adj_mat * self.weights
        egvalues, _ = np.linalg.eigh(w_mat)
        egvalues = np.flip(egvalues).reshape(1, -1) # 1 by m
        var = self.var.reshape(-1, 1) # N by 1
        egvalues = egvalues * var # N by m


    def _burden_test(self):
        """
        For a single trait Y, the burden test statistic
        (w'Z'(I-M)Y)^2 / (\hat{\sigma}^2 w'Z'(I-M)Zw)
        where w is a m by 1 vector, Z is a n by m matrix,
        M = X(X'X)^{-1}X'.
        Under the null, it follows a chisq(1) distribution.
        
        """
        burden_score_num = np.dot(np.dot(self.weights, self.half_ldr_score), self.bases.T) ** 2 # 1 by N
        burden_score = burden_score_num / (self.var * np.dot(np.dot(self.weights, self.adj_mat), self.weights.T))# 1 by N
        pvalues = chi2.sf(burden_score, 1)  
        
        return burden_score, pvalues


    def _acatv_test(self):
        half_score = np.dot(self.half_ldr_score, self.bases.T) # m by N

        common_variant_pv = chi2.sf(half_score[self.mac > 10], 1) # m1 by N
        rare_burden_score = np.sum(half_score[self.mac <= 10], axis=0) ** 2 # N by 1
        rare_burden_pv = chi2.sf(rare_burden_score, 1) # N by 1

        all_weights = self.weights ** 2 * self.maf * (1 - self.maf)
        rare_weights =  np.mean(all_weights[self.mac <= 10]) 
        common_weights = all_weights[self.mac > 10]

        test_stats = (rare_weights * np.tan((0.5 - rare_burden_pv) * np.pi) +
         np.sum(common_weights * np.tan((0.5 - common_variant_pv) * np.pi), axis=0)) 
        test_stats = test_stats / (rare_weights + np.sum(common_weights))
        
        pvalues = cauchy.sf(test_stats, loc=0, scale=1)
    
        return test_stats, pvalues


    def do_inference(self,):
        pass



def cauchy_combination(*pvalues_list):
    """
    Using cauchy combination to combine pvalues

    Parameters:
    ------------
    pvalues_list: a list of pvalues, each element should be a N by 1 array
    
    """

    pvalues = np.hstack(pvalues_list)
    stats = np.mean(np.tan((0.5 - pvalues) * np.pi), axis=1) # assuming equal weights
    cauchy_pvalues = cauchy(stats, loc=0, scale=1)
    
    return cauchy_pvalues



def remove_relatedness(ldrs, chr_preds, chr):
    mask = np.ones(chr_preds.shape[2], dtype=bool)
    mask[chr-1] = False
    loco_preds = np.sum(chr_preds[:, :, mask], axis=2)
    
    return ldrs - loco_preds



def null_model(covar, ldrs):
    inner_covar_inv = np.linalg.inv(np.dot(covar.T, covar)) # p by p
    covar_ldrs = np.dot(covar.T, ldrs) # p by r 

    return inner_covar_inv, covar_ldrs       
