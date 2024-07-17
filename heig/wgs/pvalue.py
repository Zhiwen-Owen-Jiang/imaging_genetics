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
    they share the same eigenvalues.
    p-Values for all voxels can be computed without for loop.

    Parameters:
    ------------
    score_stat: (N, ) array of score statistics
    egvalues: (m, ) array of sorted eigenvalues

    Returns:
    ---------
    pvalues: N by 1 array of pvalues

    """
    if egvalues.ndim == 1:
        egvalues = egvalues.reshape(-1, 1)
    score_stat[score_stat <= 0] = 0.0001
    # normalize eigenvalues
    n_voxels = len(score_stat)
    max_egvalue = egvalues[0]
    score_stat /= max_egvalue
    egvalues /= max_egvalue

    xmin = -len(egvalues)/(2 * score_stat)
    xmin[score_stat > np.sum(egvalues)] = -0.01
    xmax = np.ones(xmin.shape) * 0.49995

    xhat = _bisection(egvalues, score_stat, xmin, xmax)
    w = np.sqrt(2 * (xhat * score_stat - _k(xhat, egvalues)))
    w[xhat < 0] *= -1

    v = xhat * np.sqrt(_k2(xhat, egvalues))

    res = np.zeros(n_voxels)
    valid_res = abs(xhat) >= 0.0001
    res[valid_res] = norm.sf(w[valid_res] + np.log(v[valid_res] / w[valid_res]) / w[valid_res], 0, 1)
    if (~valid_res).any():
        res[~valid_res] = _handle_invalid_pvalues(score_stat[~valid_res], wcov_mat)

    return res


def _bisection(egvalues, score_stat, xmin, xmax):
    """
    Parameters:
    ------------
    egvalues: (m, ) array
    score_stat: (N, ) array
    xmin: (N, ) array
    xmax: (N, ) array

    Returns:
    ---------
    (N, ) array
    
    """
    # do iteration for 30 times to get precision ~10^-8
    for _ in range(30):
        x0 = (xmax + xmin) / 2
        k1x0 = _k1(x0, egvalues, score_stat)
        mask = k1x0 > 0
        xmax[mask] = x0[mask]
        xmin[~mask] = x0[~mask]
    return x0


def _k(x, egvalues):
    """
    Parameters:
    ------------
    x: (N, ) array
    egvalues: (m, 1) array

    Returns:
    ---------
    (N, ) array

    """
    return np.sum(np.log(1 - 2 * egvalues * x), axis=0) * -0.5


def _k1(x, egvalues, score_stat):
    """
    Parameters:
    ------------
    x: (N, ) array
    egvalues: (m, 1) array
    score_stat: (N, ) array

    Returns:
    ---------
    (N, ) array

    """
    return np.sum(egvalues / (1 - 2 * egvalues * x), axis=0) - score_stat


def _k2(x, egvalues):
    """
    Parameters:
    ------------
    x: (N, ) array
    egvalues: (m, 1) array

    Returns:
    ---------
    (N, ) array

    """
    return np.sum(egvalues ** 2 / (1 - 2 * egvalues * x) ** 2, axis=0) * 2


def _handle_invalid_pvalues(score_stat, wcov_mat):
    """
    Dealing with the case the saddle fails.

    Parameters:
    ------------
    score_stat: (N, ) array
    wcov_mat: (m, m) array

    Returns:
    ---------
    pvalues: (N, ) array
    
    """
    c1 = np.sum(np.diag(wcov_mat))
    wcov_mat = np.dot(wcov_mat, wcov_mat)
    c2 = np.sum(np.diag(wcov_mat))
    wcov_mat = np.dot(wcov_mat, wcov_mat)
    c4 = np.sum(np.diag(wcov_mat))

    score_stat = (score_stat - c1) / np.sqrt(2 * c2)
    l = c2 ** 2 / c4
    pvalues = chi2.sf(score_stat * np.sqrt(2 * l) + l, l)

    return pvalues