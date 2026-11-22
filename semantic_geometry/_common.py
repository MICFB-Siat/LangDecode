import json, re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse
from scipy.stats import gamma, rankdata
from langdecoder.paths import DATA, OUT, LPPC, N_CORTEX, LANGS

PAIRS = (("CN", "EN"), ("CN", "FR"), ("EN", "FR"))
N_ANCHOR, N_STORY, TR = 24, 9, 2.0
EVENTS = DATA / "events"
FEATURES = DATA / "features"
NEIGHBOURS = DATA / "atlas" / "searchlight_8mm_csr.npz"
COHORT = LPPC / "participants.json"
OUT3 = OUT / "semantic_geometry"


def cohort():
    return json.loads(COHORT.read_text())["languages"]


def events(lang):
    return pd.read_csv(EVENTS / f"lppc_{lang}.csv")


def run_files(subject, root=LPPC, pattern="*.dtseries.nii"):
    files = sorted((root / f"sub-{subject}").glob(pattern), key=lambda f: int(re.search(r"run-(\d+)", f.name).group(1)))
    if len(files) != N_STORY:
        raise RuntimeError(f"{subject}: {len(files)} runs")
    return files


def spm_hrf(tr=TR, oversampling=20, duration=32.0):
    t = np.arange(0, duration, tr / oversampling); h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / max(h.sum(), 1e-12)


def event_design(ev, n_tr, tr=TR, oversampling=20):

    n_high, dt = n_tr * oversampling, tr / oversampling; high = np.zeros((n_high, len(ev)))
    for k, row in enumerate(ev.itertuples(index=False)):
        a = max(0, int(np.floor(row.glm_onset / dt))); b = min(n_high, max(a + 1, int(np.ceil(row.glm_offset / dt))))
        if a < n_high: high[a:b, k] = 1.0
    hrf = spm_hrf(tr, oversampling)
    x = np.stack([np.convolve(high[:, k], hrf)[:n_high] for k in range(len(ev))], 1)[np.arange(n_tr) * oversampling]
    return x / np.maximum(np.abs(x).max(0, keepdims=True), 1e-12)


def dct_basis(n_tr, tr=TR, cutoff=128.0):
    t = np.arange(n_tr, dtype=float); n = int(np.floor(2 * n_tr * tr / cutoff))
    return np.column_stack([np.ones(n_tr)] + [np.cos(np.pi * (t + .5) * k / n_tr) for k in range(1, n + 1)])


def lss_weights(ev, n_tr):

    x = event_design(ev, n_tr); nuis = dct_basis(n_tr); allx = x.sum(1); w = np.empty((len(ev), n_tr), np.float32)
    for k in range(len(ev)):
        w[k] = np.linalg.pinv(np.column_stack([x[:, k], allx - x[:, k], nuis]), rcond=1e-8)[0]
    return w


def zscore_time(x, valid):
    out = np.zeros_like(x, dtype=np.float32); c = x[:, valid]; sd = c.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
    out[:, valid] = np.nan_to_num((c - c.mean(0, keepdims=True)) / sd); return out


def load_run(path, phase_seed=None):

    x = np.asarray(nib.load(str(path)).dataobj[:, :N_CORTEX], np.float32)
    if phase_seed is not None:
        rng = np.random.default_rng(phase_seed); mu = x.mean(0); f = np.fft.rfft(x - mu, axis=0)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, f.shape[0])); ph[0] = 1
        if len(x) % 2 == 0: ph[-1] = 1
        x = (np.fft.irfft(f * ph[:, None], n=len(x), axis=0) + mu).astype(np.float32)
    valid = np.isfinite(x).all(0) & (np.nanstd(x, 0) > 1e-6)
    return x, valid


def story_betas(path, ev, rows=None, phase_seed=None):

    x, valid = load_run(path, phase_seed); w = lss_weights(ev, len(x))
    if rows is not None: w = w[rows]
    b = (w @ zscore_time(x, valid)).astype(np.float32); b[:, ~valid] = 0.0
    return b, valid


def lppc_anchors():

    tabs = {L: events(L) for L in LANGS}; out = {}
    for s in range(1, N_STORY + 1):
        n = min(int((t.story == s).sum()) for t in tabs.values())
        out[s] = np.rint(np.linspace(0, n - 1, N_ANCHOR)).astype(int)
    return out


def corr_rdm(p):
    p = p - p.mean(1, keepdims=True); p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
    return (1 - p @ p.T).astype(np.float32)


def cosine_rdm_vec(e):
    e = e / np.linalg.norm(e, axis=1, keepdims=True); return (1 - e @ e.T)[np.triu_indices(len(e), 1)]


def neighbourhoods():
    z = np.load(NEIGHBOURS); return sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def searchlight_rdms(beta, member, valid, min_vertices=20):

    p = beta * valid[None].astype(np.float32); iu = np.triu_indices(len(p), 1)
    cnt = np.asarray(member @ valid.astype(np.float32)).ravel(); s = np.asarray(member @ p.T, np.float32)
    e = np.asarray(member @ np.square(p).T, np.float32); cross = np.asarray(member @ (p[iu[0]] * p[iu[1]]).T, np.float32)
    n = np.maximum(cnt, 1)[:, None]; cov = cross - s[:, iu[0]] * s[:, iu[1]] / n
    va = e[:, iu[0]] - s[:, iu[0]] ** 2 / n; vb = e[:, iu[1]] - s[:, iu[1]] ** 2 / n
    d = 1.0 - np.clip(cov / np.sqrt(np.maximum(va * vb, 1e-20)), -1, 1)
    good = (cnt >= min_vertices) & np.all(np.isfinite(d), 1) & np.all((va > 1e-12) & (vb > 1e-12), 1); d[~good] = np.nan
    return d.astype(np.float32)


def rank_rows(x):

    x = np.asarray(x, np.float64); out = np.full(x.shape, np.nan, np.float32); good = np.all(np.isfinite(x), -1)
    r = rankdata(x[good], axis=-1); r -= r.mean(-1, keepdims=True); out[good] = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    return out


def fz(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def edge_permutation(order, n=N_ANCHOR):

    m = np.zeros((n, n), int); iu = np.triu_indices(n, 1); m[iu] = np.arange(len(iu[0])); m = m + m.T
    return m[np.ix_(order, order)][iu]


def cmds(d, k=3):
    n = len(d); j = np.eye(n) - 1 / n; b = -0.5 * j @ (d ** 2) @ j; w, v = np.linalg.eigh(b); o = np.argsort(w)[::-1][:k]
    return v[:, o] * np.sqrt(np.maximum(w[o], 0))


def bh_fdr(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.clip(q, 0, 1); return out
