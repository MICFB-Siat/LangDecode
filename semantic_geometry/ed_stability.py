import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from _common import OUT3, LANGS, PAIRS, N_STORY, N_CORTEX, cohort, rank_rows, fz, corr_rdm

O = OUT3 / "stability"; W, STEP, N_NULL = 30, 10, 1000


def spear(a, b):
    ra, rb = rankdata(a) - (len(a) + 1) / 2, rankdata(b) - (len(b) + 1) / 2; return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def story_splits():
    c = cohort(); G = {}
    for L in LANGS:
        tot = 0
        for s in c[L]: tot = tot + np.load(OUT3 / "searchlight" / f"{L}_{s}.npz")["rdm"].astype(np.float32)
        G[L] = np.stack([rank_rows(tot[k] / len(c[L])) for k in range(N_STORY)])
    rng, rows = np.random.default_rng(20260928), []
    splits = [rng.permutation(N_STORY) for _ in range(100)]
    for a, b in PAIRS:
        z = fz(np.einsum("sve,sve->sv", G[a], G[b]))
        for sp in splits:
            ma, mb = np.tanh(np.nanmean(z[sp[:4]], 0)), np.tanh(np.nanmean(z[sp[4:]], 0)); ok = np.isfinite(ma) & np.isfinite(mb)
            rows.append({"pair": f"{a}-{b}", "spatial_spearman": spearmanr(ma[ok], mb[ok])[0]})
    pd.DataFrame(rows).to_csv(O / "story_split_stability.csv", index=False, float_format="%.5f")


def narrative_windows():
    real = {L: np.load(OUT3 / "betas" / f"group_{L}.npz", allow_pickle=True) for L in LANGS}
    null = {L: np.load(OUT3 / "betas_prnull" / f"group_{L}.npz", allow_pickle=True) for L in LANGS}
    common = set.intersection(*(set(real[L]["sentence_index"]) for L in LANGS))
    rows_of = {L: np.flatnonzero(np.isin(real[L]["sentence_index"], list(common))) for L in LANGS}; story = real["CN"]["story"][rows_of["CN"]]
    R = {(v, L, s): corr_rdm(src[L]["beta"][rows_of[L]].astype(np.float32)[story == s]) for v, src in (("real", real), ("null", null)) for L in LANGS for s in range(1, N_STORY + 1)}
    up = lambda d, st: d[st:st + W, st:st + W][np.triu_indices(W, 1)]; rng = np.random.default_rng(100 + W); out = []
    for s in range(1, N_STORY + 1):
        for a, b in PAIRS:
            A, B = R[("real", a, s)], R[("real", b, s)]; n = len(A); starts = np.arange(0, n - W + 1, STEP); lo = max(10, int(np.ceil(0.1 * n)))
            shifted = [B[np.ix_(p, p)] for p in (np.roll(np.arange(n), k) for k in rng.integers(lo, n - lo + 1, N_NULL))]
            for st in starts:
                obs = spear(up(A, st), up(B, st)); nl = [spear(up(A, st), up(Bk, st)) for Bk in shifted]
                out.append({"story": s, "pair": f"{a}-{b}", "progress": 100 * (st + W / 2) / n, "window_rsa": obs, "circular_null_mean": float(np.mean(nl)),
                            "circular_null_97.5": float(np.percentile(nl, 97.5)), "design_null_rsa": spear(up(R[("null", a, s)], st), up(R[("null", b, s)], st))})
    d = pd.DataFrame(out); d.to_csv(O / "narrative_windows.csv", index=False, float_format="%.5f")
    d["bin"] = np.minimum((d.progress // 10).astype(int), 9) * 10 + 5
    g = d.groupby(["pair", "bin"]).agg(n=("window_rsa", "size"), mean=("window_rsa", "mean"), sem=("window_rsa", "sem"),
                                       design_null=("design_null_rsa", "mean"), circular_null=("circular_null_mean", "mean")).reset_index()
    g[g.n >= 5].to_csv(O / "narrative_windows_binned.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    O.mkdir(parents=True, exist_ok=True); story_splits(); narrative_windows()
