import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon
from joblib import Parallel, delayed
from langdecoder.paths import ATLAS, N_CORTEX
from _common import OUT3, EVENTS, LANGS, PAIRS, N_STORY, cohort, run_files, load_run

O = OUT3 / "relational_alignment"; N_BINS = 20


def destrieux_average():
    lab = np.load(ATLAS, allow_pickle=True)["destrieux"].astype(int)[:N_CORTEX]; ids = np.unique(lab[lab > 0]); r, c, w = [], [], []
    for j, p in enumerate(ids):
        v = np.flatnonzero(lab == p); r += v.tolist(); c += [j] * len(v); w += [1.0 / len(v)] * len(v)
    return sparse.csr_matrix((w, (r, c)), shape=(N_CORTEX, len(ids)), dtype=np.float32)


def windows(lang):
    return pd.read_csv(EVENTS / f"lppc_windows10s_{lang}.csv").reset_index(drop=True)


def bins(values, win):

    out = []
    for s in range(1, N_STORY + 1):
        w = win[win.story == s]; b = np.minimum((w.relative_story_position.to_numpy() * N_BINS).astype(int), N_BINS - 1)
        out += [values[w.index.to_numpy()[b == k]].mean(0) for k in range(N_BINS)]
    return np.stack(out).reshape(N_STORY, N_BINS, -1)


def participant_bins(lang, sub, avg):
    win, files = windows(lang), run_files(sub); resp = np.empty((len(win), avg.shape[1]), np.float32)
    for s in range(1, N_STORY + 1):
        x, _ = load_run(files[s - 1]); sd = x.std(0, keepdims=True); sd[sd < 1e-6] = 1
        pt = np.asarray(np.nan_to_num((x - x.mean(0, keepdims=True)) / sd) @ avg)
        for i, r in win[win.story == s].iterrows(): resp[i] = pt[int(r.bold_start_tr):int(r.bold_end_tr)].mean(0)
    return bins(resp, win)


def rsm(p):
    p = p - p.mean(1, keepdims=True); p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12); return np.clip(p @ p.T, -1, 1)


def alignment(a, b):
    n = len(a); g = np.full((n, n), np.nan); idx = np.arange(n)
    for i in range(n):
        for j in range(n):
            k = (idx != i) & (idx != j); x = a[i, k] - a[i, k].mean(); y = b[j, k] - b[j, k].mean()
            d = np.linalg.norm(x) * np.linalg.norm(y); g[i, j] = x @ y / d if d > 1e-12 else np.nan
    return g


def circular_excess(g):
    n = len(g); diag = np.array([np.nanmean(g[np.arange(n), (np.arange(n) + k) % n]) for k in range(n)]); return diag[0] - diag[1:].mean()


def main():
    O.mkdir(parents=True, exist_ok=True); c, avg = cohort(), destrieux_average()
    P = {L: dict(zip(c[L], Parallel(n_jobs=8)(delayed(participant_bins)(L, s, avg) for s in c[L]))) for L in LANGS}
    group = {L: np.mean(list(P[L].values()), 0) for L in LANGS}; rows = []
    for a, b in PAIRS:
        for src, tgt in ((a, b), (b, a)):
            for sub, x in P[tgt].items():
                e = [circular_excess(alignment(rsm(group[src][s]), rsm(x[s]))) for s in range(N_STORY)]
                rows.append({"pair": f"{a}-{b}", "source": src, "target": tgt, "participant": sub, "circular_diagonal_excess": float(np.mean(e))})
    d = pd.DataFrame(rows); d.to_csv(O / "participant_alignment.csv", index=False, float_format="%.6f")
    pd.DataFrame({p: pd.Series(d[d.pair == p].circular_diagonal_excess.to_numpy()) for p in d.pair.unique()}).to_csv(O / "fig3b_prism.csv", index=False, float_format="%.5f")
    st = [{"pair": p, "n": len(v), "median": float(np.median(v)), "wilcoxon_one_sided_p": float(wilcoxon(v, alternative="greater").pvalue)}
          for p, v in ((p, d[d.pair == p].circular_diagonal_excess.to_numpy()) for p in d.pair.unique())]
    pd.DataFrame(st).to_csv(O / "fig3b_stats.csv", index=False); print(pd.DataFrame(st))


if __name__ == "__main__":
    main()
