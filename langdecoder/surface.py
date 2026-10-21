import numpy as np
from .paths import ATLAS

_PARCEL = None


def parcel_labels(key="parcel"):
    global _PARCEL
    if _PARCEL is None:
        _PARCEL = np.load(ATLAS, allow_pickle=True)[key].astype(int)
    return _PARCEL


def parcel_mean(v, n_parcels=400):

    p = parcel_labels(); v = np.asarray(v, float); ok = np.isfinite(v) & (p > 0)
    s = np.bincount(p[ok], weights=v[ok], minlength=n_parcels + 1)[1:]
    c = np.bincount(p[ok], minlength=n_parcels + 1)[1:]
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)
