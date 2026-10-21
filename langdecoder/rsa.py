import numpy as np
from scipy.stats import spearmanr


def corr_distance(x):

    x = np.asarray(x, float); x = x - x.mean(1, keepdims=True); x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.clip(x @ x.T, -1, 1)


def upper(m, k=1):
    return m[np.triu_indices(len(m), k)]


def rdm_spearman(a, b):
    return float(spearmanr(upper(a), upper(b))[0])


def fisher_mean(r, axis=0):
    return np.tanh(np.nanmean(np.arctanh(np.clip(r, -0.999999, 0.999999)), axis=axis))
