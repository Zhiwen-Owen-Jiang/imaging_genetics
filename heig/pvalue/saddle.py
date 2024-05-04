import numpy as np
from scipy.stats import norm, chi2


"""
Credits to Xihao Li and the STAAR developers for the following functions
https://github.com/xihaoli/STAAR

"""


def saddle(score_stat, egvalues, wcov_mat):
    """
    Saddlepoint approximation.
    Score statistics have been normalized such that
    they share the same eigen values.
    Hopefully, this will speed up the calculation. 

    Parameters:
    ------------
    score_stat: N by 1 array of score statistics
    egvalues: m by 1 array of eigenvalues

    Returns:
    ---------
    pvalues: N by 1 array of pvalues

    """
    # normalize eigenvalues
    n_voxels = len(score_stat)
    max_egvalue = np.max(egvalues)
    score_stat /= max_egvalue
    egvalues /= max_egvalue

    xmin = -len(egvalues)/(2 * score_stat)
    xmin[score_stat > np.sum(egvalues)] = -0.01
    xmax = 0.49995

    xhat = np.zeros(n_voxels)
    for i in range(n_voxels):
        xhat[i] = _bisection(egvalues, score_stat[i], xmin[i], xmax)
    w = np.sqrt(2 * (xhat * score_stat - _k(xhat, egvalues)))
    w[xhat < 0] *= -1

    v = xhat * np.sqrt(_k2(xhat, egvalues))
    
    res = np.zeros(n_voxels)
    valid_res = abs(xhat) >= 0.0001
    res[valid_res] = norm.sf(w[valid_res] + np.log(v[valid_res] / w[valid_res]) / w[valid_res], 0, 1)
    res[~valid_res] = _handle_invalid_pvalues(score_stat[~valid_res], wcov_mat)
    
    return res



def _bisection(egvalues, score_stat, xmin, xmax):
    while np.abs(xmax - xmin) > 10**-8:
        x0 = (xmax + xmin) / 2
        k1x0 = _k1(x0, egvalues, score_stat)
        if k1x0 == 0:
            return x0
        elif k1x0 > 0:
            xmax = x0
        else:
            xmin = x0

    return x0



def _k(x, egvalues):
    """
    Parameters:
    ------------
    x: (N, ) array
    egvalues: (m, ) array

    Returns:
    ---------
    (N, ) array
    
    """
    return np.sum(np.log(1 - 2 * egvalues.reshape(-1, 1) * x), axis=0) * -0.5
    


def _k1(x, egvalues, score_stat):
    """
    Parameters:
    ------------
    x: (N, ) array
    egvalues: (m, ) array
    score_stat: (N, ) array

    Returns:
    ---------
    (N, ) array
    
    """
    return np.sum(egvalues / (1 - 2 * egvalues.reshape(-1, 1) * x), axis=0) - score_stat


    
def _k2(x, egvalues):
    """
    Parameters:
    ------------
    x: (N, ) array
    egvalues: (m, ) array

    Returns:
    ---------
    (N, ) array
    
    """
    return np.sum(egvalues ** 2 / (1 - 2 * egvalues.reshape(-1, 1) * x) ** 2, axis=0) * 2



def _handle_invalid_pvalues(score_stat, wcov_mat):
    c1 = np.sum(np.diag(wcov_mat))
    wcov_mat = np.dot(wcov_mat, wcov_mat)
    c2 = np.sum(np.diag(wcov_mat))
    wcov_mat = np.dot(wcov_mat, wcov_mat)
    c4 = np.sum(np.diag(wcov_mat))

    score_stat = (score_stat - c1) / np.sqrt(2 * c2)
    l = c2 ** 2 / c4
    pvalues = chi2.sf(score_stat * np.sqrt(2 * l) + l, l)
    
    return pvalues
