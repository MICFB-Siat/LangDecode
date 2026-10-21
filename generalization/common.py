import numpy as np
from scipy.stats import rankdata, spearmanr
from langdecoder.paths import OUT, G1 as G1_FILE, N_CORTEX
from langdecoder.surface import parcel_labels, parcel_mean

PAIRS = ("CN_EN", "CN_FR", "EN_FR")
DIRECTIONS = ("cn2en", "en2cn", "cn2fr", "fr2cn", "en2fr", "fr2en")
TRAIN_STORIES = 8


def pair_of(direction):
    a, b = direction.upper().split("2"); return "_".join(sorted([a, b]))


def g1_vertex():

    return np.load(G1_FILE).astype(float)


def shared_map_vertex(pair=None):

    maps = [np.load(OUT / "generalization" / f"shared_map_train_{p}.npy").astype(float) for p in ((pair,) if pair else PAIRS)]
    v = np.mean(maps, 0); return np.where(np.isfinite(v), v, np.nanmean(v))


def thirds(parcel_values):

    n = np.isfinite(parcel_values).sum(); return np.minimum((rankdata(parcel_values) / n * 3).astype(int), 2)


def tier_vertex_labels(parcel_tier):

    p = parcel_labels(); out = np.zeros(N_CORTEX, np.int16); m = p > 0; out[m] = parcel_tier[p[m] - 1] + 1; return out


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b); return float(spearmanr(a[m], b[m])[0])


def load_region_2afc(direction):

    z = np.load(OUT / "generalization" / f"single_region_{direction}.npz"); v = np.full(400, np.nan); v[z["parcel_ids"] - 1] = z["parcel_2afc"]; return v
