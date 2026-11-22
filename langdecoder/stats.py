import numpy as np


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i])); out[i] = run
    return out


def stars(p):
    return "****" if p <= 1e-4 else "***" if p <= 1e-3 else "**" if p <= 1e-2 else "*" if p <= 0.05 else "n.s."
