"""
steps:
0. partition the region to some folds
1. flip SNPs, compute MAF and filter MAF with a threshold
2. get weight for each variant
3. for each fold, get the subregion
    3.0. get threshold SCANG_O_Thres
    3.1. get the maximum across folds, get quantile, and -log()
    3.2. do search using the threshold
4. filter region

"""
import numpy as np
from scipy.stats import chi2
from staar import cauchy_combination


def threshold(lmax, lmin, steplength, weight_B, weight_S, filter):
    pass

def maxO(p, Lmax, Lmin, steplength, x,
         weights_B, weights_S, Cov,
         fitler, times):
    """
    x: (times, n_variants)
    weights_B: (n_variants, n_weights)
    
    """
    n_variants, n_weights = weights_B.shape

    Covw_B = np.zeros((n_weights, n_variants, n_variants))
    Covw_S = np.zeros((n_weights, n_variants, n_variants))
    for i in range(n_weights):
        Covw_B[i] = Cov * weights_B[:, i] * weights_B[:, i].T
        Covw_S[i] = Cov * weights_S[:, i] * weights_S[:, i].T
    Covw2_S = Covw_S ** 2

    filtervalue_b = chi2.isf(filter, 1)
    lengthnum = int((Lmax-Lmin)/steplength) + 1

    for ij in range(lengthnum):
        window_i = Lmin + ij*steplength
        filtervalue = (chi2.isf(filter, window_i) - window_i) / np.sqrt(2*window_i)

        sum0_s = np.dot(weights_S[:window_i].T ** 2, x[:, :window_i].T ** 2) # (n_weights, times)
        sum0_b = np.dot(weights_B[:window_i].T, x[:, :window_i].T) # (n_weights, times)

        c1 = np.zeros(n_weights)
        for i in range(n_weights):
            c1[i] = np.sum(np.diag(Covw_S[i])[:window_i])

        if window_i > 1:
            w = np.sum(Covw_B[:, :window_i, :window_i], axis=(1,2))
            c2 = np.sum(Covw2_S[:, :window_i, :window_i], axis=(1,2))
        else:
            w = Covw_B[:, 0, 0]
            c2 = Covw2_S[:, 0, 0]

        ## Burden
        sumx_b = sum0_b ** 2 / w
        ## SKAT
        tstar = (sum0_s - c1) / np.sqrt(2 * c2)



def scang_search(G, X, sigma, residuals,
                 threshold_o, threshold_s, threshold_b, 
                 Lmax, Lmin, steplength, begid,
                 weights_B, weights_S, filter, f):
    n_variants, n_weights = weights_B.shape

    u_score = np.dot(G.T, residuals) # (n_variants, )
    tX_G = np.dot(X.T, G)
    Cov = np.dot(G.T, G) - np.dot(np.dot(tX_G.T, np.linalg.inv(np.dot(X.T, X))), tX_G)
    Cov *= sigma**2

    Covw_B = np.zeros((n_weights, n_variants, n_variants))
    Covw_S = np.zeros((n_weights, n_variants, n_variants))
    for i in range(n_weights):
        Covw_B[i] = Cov * weights_B[:, i] * weights_B[:, i].T
        Covw_S[i] = Cov * weights_S[:, i] * weights_S[:, i].T

    lengthnum = int((Lmax-Lmin)/steplength) + 1

    candidate = list()
    candidate_s = list()
    candidate_b = list()
    summax = -100000.0
    summax_s = -100000.0
    summax_b = -100000.0
    filtervalue_b = chi2.isf(filter, 1)

    for ij in range(lengthnum):
        window_i = Lmin + ij*steplength
        filtervalue = (chi2.isf(filter, window_i) - window_i) / np.sqrt(2*window_i)
        
        ## Q_burden and Q_skat
        sum0_s = np.dot(weights_S[:window_i].T ** 2, u_score[:window_i] ** 2) # (n_weights, )
        sum0_b = np.dot(weights_B[:window_i].T, u_score[:window_i]) # (n_weights, )

        ## moments
        w1 = np.zeros(n_weights)
        for i in range(n_weights):
            w1[i] = np.sum(np.diag(Covw_S[i])[:window_i])

        w2 = np.zeros(n_weights)
        ii, jj = np.triu_indices(window_i, 1)
        w2 += np.sum(Covw_S[:, ii, ii] * Covw_S[:, jj, jj] - Covw_S[:, ii, jj] * Covw_S[:, jj, ii], axis=1)
        # for i in range(n_weights):
        #     for ii in range(window_i-1):
        #         for jj in range(ii+1, window_i):
        #             w2[i] += (Covw_S[i, ii, ii] * Covw_S[i, jj, jj] -
        #                       Covw_S[i, ii, jj] * Covw_S[i, jj, ii])
        
        c1 = w1
        c2 = w1 ** 2 - 2 * w2
        if window_i > 1:
            w = np.sum(Covw_B[:, :window_i, :window_i], axis=(1,2))
        else:
            w = Covw_B[:, 0, 0] # (n_weights, )

        ## burden
        sumx_b = sum0_b ** 2 / w

        ## SKAT
        tstar = (sum0_s - c1) / np.sqrt(2 * c2)
        ## Here we only keep the test statistics but not compute the exact p-value
        sump_s = np.zeros(n_weights)
        sump_s[tstar > filtervalue] = sum0_s[tstar > filtervalue]

        ## Here sum0_s should be greater than a statistic
        if ((sumx_b > filtervalue_b) | (sum0_s > filtervalue)).any():
            sump_b = chi2.ppf(sumx_b, 1)
            sump_s[tstar <= filtervalue] = sum0_s[tstar <= filtervalue]

            CCT_p = np.zeros(2 * n_weights)
            CCT_p[range(0, n_weights, 2)] = sump_s
            CCT_p[range(1, n_weights, 2)] = sump_b
            CCT_ps = sump_s
            CCT_pb = sump_b

            ## SCANG_O
            sump_o = -np.log(cauchy_combination(CCT_p, np.ones(2 * n_weights)))
            ## SCANG_S
            sump_os = -np.log(cauchy_combination(CCT_ps, np.ones(n_weights)))
            ## SCANG_B
            sump_ob = -np.log(cauchy_combination(CCT_pb, np.ones(n_weights)))

            ## signal regions
            if sump_o > threshold_o:
                candidate.append(np.array([sump_o, 1, window_i, 0]))
            if sump_os > threshold_s:
                candidate_s.append(np.array([sump_os, 1, window_i, 0]))
            if sump_ob > threshold_b: 
                candidate_b.append(np.array([sump_ob, 1, window_i, 0]))

            ## top 1 region
            if sump_o > summax:
                summax = sump_o
                candidatemax = np.array([sump_o, 1 + begid - 1, window_i + begid - 1, 0])
            if sump_os > summax_s:
                summax_s = sump_os
                candidatemax_s = np.array([sump_os, 1 + begid - 1, window_i + begid - 1, 0])
            if sump_ob > summax_b:
                summax_b = sump_ob
                candidatemax_b = np.array([sump_ob, 1 + begid - 1, window_i + begid - 1, 0])

        for j in range(1, n_variants - window_i + 1):
            w1 = w1 - Covw_S[:, j-1, j-1] + Covw_S[:, j+window_i-1, j+window_i-1]

            for kk in range(1, window_i):
                w2 = w2 - Covw_S[:, j-1, j-1]*Covw_S[:, j-1+kk, j-1+kk] - Covw_S[:, j-1, j-1+kk]*Covw_S[j-1+kk, j-1]
                w2 = w2 + Covw_S[:, j+window_i-1, j+window_i-1]*Covw_S[:, j-1+kk, j-1+kk] - Covw_S[:, j+window_i-1, j-1+kk]*Covw_S[j-1+kk, j+window_i-1]

            c1 = w1
            c2 = w1 ** 2 - 2 * w2

            if window_i > 1:
                w = w - np.sum(Covw_B[:, j-1, j: j+window_i-1]) - np.sum(Covw_B[:, j:j+window_i-1, j-1]) - Covw_B[:, j-1, j-1]
                w = w + np.sum(Covw_B[:, j+window_i-1, j:j+window_i-1]) + np.sum(Covw_B[:, j:j+window_i-1, j+window_i-1]) + Covw_B[:, j+window_i-1, j+window_i-1]
            else:
                w = w - Covw_B[j-1, j-1] + Covw_B[j+window_i-1, j+window_i-1]

            sump_b = 1
            sump_s = 1
            
            ## burden
            sum0_b = sum0_b - u_score[j-1]*weights_B[j-1] + u_score[j+window_i-1]*weights_B[j+window_i-1]
            sumx_b = sum0_b ** 2 / w

            ## SKAT
            sum0_s = sum0_s - u_score[j-1] ** 2 * weights_S[j-1] ** 2 + u_score[j+window_i-1] ** 2 * weights_S[j+window_i-1] ** 2
            tstar = (sum0_s - c1) / np.sqrt(2 * c2)
            ## Here we only keep the test statistics but not compute the exact p-value
            sump_s[tstar > filtervalue] = sum0_s[tstar > filtervalue]

            if ((sumx_b > filtervalue_b) | (sump_s < filter)).any():
                ## burden
                sump_b = chi2.ppf(sumx_b, 1)

                ## SKAT
                ## Here we only keep the test statistics but not compute the exact p-value
                sump_s[tstar <= filtervalue] = sum0_s[tstar <= filtervalue]

                CCT_p[range(0, n_weights, 2)] = sump_s
                CCT_p[range(1, n_weights, 2)] = sump_b
                CCT_ps = sump_s
                CCT_pb = sump_b

                ## SCANG_O
                sump_o = -np.log(cauchy_combination(CCT_p, np.ones(2 * n_weights)))
                ## SCANG_S
                sump_os = -np.log(cauchy_combination(CCT_ps, np.ones(n_weights)))
                ## SCANG_B
                sump_ob = -np.log(cauchy_combination(CCT_pb, np.ones(n_weights)))

                ## signal regions
                if sump_o > threshold_o:
                    candidate.append(np.array([sump_o, j + 1, j + window_i, 0]))
                if sump_os > threshold_s:
                    candidate_s.append(np.array([sump_os, j + 1, j + window_i, 0]))
                if sump_ob > threshold_b: 
                    candidate_b.append(np.array([sump_ob, j + 1, j + window_i, 0]))

                if sump_o > summax:
                    summax = sump_o
                    candidatemax = np.array([sump_o, j + 1 + begid - 1, j + window_i + begid - 1, 0])
                if sump_os > summax_s:
                    summax_s = sump_os
                    candidatemax_s = np.array([sump_os, j + 1 + begid - 1, j + window_i + begid - 1, 0])
                if sump_ob > summax_b:
                    summax_b = sump_ob
                    candidatemax_b = np.array([sump_ob, j + 1 + begid - 1, j + window_i + begid - 1, 0])

    ## SCANG_O
    candidate = np.vstack(candidate)
    candidate = candidate[np.argsort(-candidate[:, 0])]
    loc_left = 0
    loc_right = 0
    for ii in range(len(candidate)-1):
        if candidate[ii, 3] < 1:
            for jj in range(ii+1, len(candidate)):
                if candidate[ii, 1] < candidate[jj, 1]:
                    loc_left = candidate[jj, 1]
                else:
                    loc_left = candidate[ii, 1]
                if candidate[ii, 2] < candidate[jj, 2]:
                    loc_right = candidate[ii, 2]
                else:
                    loc_right = candidate[jj, 2]
                if loc_right > loc_left - 1:
                    if (loc_right - loc_left + 1) / (candidate[jj, 2] - candidate[jj, 1] + 1) > f:
                        candidate[jj, 3] = 1
    
    res = list()
    for kk in range(len(candidate)):
        if candidate[kk, 3] < 1:
            res.append(candidate[kk, 0], candidate[kk, 1] + begid - 1, candidate[kk, 2] + begid - 1, 0)

    ## SCANG_S
    candidate_s = np.vstack(candidate_s)
    candidate_s = candidate_s[np.argsort(-candidate_s[:, 0])]
    loc_left_s = 0
    loc_right_s = 0
    for ii in range(len(candidate_s)-1):
        if candidate_s[ii, 3] < 1:
            for jj in range(ii+1, len(candidate_s)):
                if candidate_s[ii, 1] < candidate_s[jj, 1]:
                    loc_left_s = candidate_s[jj, 1]
                else:
                    loc_left_s = candidate_s[ii, 1]
                if candidate_s[ii, 2] < candidate_s[jj, 2]:
                    loc_right_s = candidate_s[ii, 2]
                else:
                    loc_right_s = candidate_s[jj, 2]
                if loc_right_s > loc_left_s - 1:
                    if (loc_right_s - loc_left_s + 1) / (candidate_s[jj, 2] - candidate_s[jj, 1] + 1) > f:
                        candidate_s[jj, 3] = 1

    res_s = list()
    for kk in range(len(candidate_s)):
        if candidate_s[kk, 3] < 1:
            res_s.append(candidate_s[kk, 0], candidate_s[kk, 1] + begid - 1, candidate_s[kk, 2] + begid - 1, 0)

    ## SCANG_B
    candidate_b = np.vstack(candidate_b)
    candidate_b = candidate_b[np.argsort(-candidate_b[:, 0])]
    loc_left_b = 0
    loc_right_b = 0
    for ii in range(len(candidate_b)-1):
        if candidate_b[ii, 3] < 1:
            for jj in range(ii+1, len(candidate_b)):
                if candidate_b[ii, 1] < candidate_b[jj, 1]:
                    loc_left_b = candidate_b[jj, 1]
                else:
                    loc_left_b = candidate_b[ii, 1]
                if candidate_b[ii, 2] < candidate_b[jj, 2]:
                    loc_right_b = candidate_b[ii, 2]
                else:
                    loc_right_b = candidate_b[jj, 2]
                if loc_right_b > loc_left_b - 1:
                    if (loc_right_b - loc_left_b + 1) / (candidate_b[jj, 2] - candidate_b[jj, 1] + 1) > f:
                        candidate_b[jj, 3] = 1

    res_b = list()
    for kk in range(len(candidate_b)):
        if candidate_b[kk, 3] < 1:
            res_b.append(candidate_b[kk, 0], candidate_b[kk, 1] + begid - 1, candidate_b[kk, 2] + begid - 1, 0)
            
    results = {'res_o': res, 'resmost_o': candidatemax,
               'res_s': res_s, 'resmost_s': candidatemax_s,
               'res_b': res_b, 'resmost_b': candidatemax_b
               }
    
    return results


def search(u_scores, wcov_mat, Lmin, Lmax, 
           threshold_o, threshold_s, threshold_b, ):
    """
    u_scores: normalized weighted scores WZ'(I-M)\Xi\Phi/var (m, N) 
    wcov_mat: weighted covariance matrix WCovW (m, m), all voxels share the same one
    
    """
    












