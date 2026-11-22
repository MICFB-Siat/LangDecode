import sys, json, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "generalization"))
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from langdecoder.paths import OUT
from langdecoder.surface import parcel_mean
from langdecoder.spin import SpinNull, spun_parcel_maps
from fig4_gradient_correspondence import cifti_axis
from common import g1_vertex, shared_map_vertex

NB = 20


def table(M, v):
    b = np.minimum((rankdata(v) / len(v) * NB).astype(int), NB - 1); return np.array([[row[b == k].mean() for k in range(NB)] for row in M])


def position(tab):
    w = tab / (tab.sum(1, keepdims=True) + 1e-12); return (w * (np.arange(NB) + 0.5)).sum(1) / NB


def main():
    Z = np.load(OUT / "extended_data" / "neurosynth_term_parcels.npz"); T = [str(t) for t in Z["terms"]]; M = np.clip(np.nan_to_num(Z["parcels"]), 0, None)
    g1v, gv = g1_vertex(), shared_map_vertex(); G1, GEO = parcel_mean(g1v), parcel_mean(gv); axis = cifti_axis()
    TG, TS = table(M, G1), table(M, GEO); pG, pS = position(TG), position(TS)
    for name, tab in (("g1", TG), ("semantic", TS)): pd.DataFrame(tab, index=T, columns=[f"{5 * k}-{5 * k + 5}%" for k in range(NB)]).to_csv(OUT / "extended_data" / f"neurosynth_{name}_bins.csv")
    null = np.array([position(table(M, s)) - pG for s in spun_parcel_maps(gv, SpinNull(axis, 1000, seed=20261008))]); c0 = null.mean(0); d = pS - pG
    p = (1 + (np.abs(null - c0) >= np.abs(d - c0)).sum(0)) / (1 + len(null)); q = multipletests(p, method="fdr_bh")[1]
    rg, rs = rankdata(np.nan_to_num(g1v, nan=np.nanmean(g1v))) / len(g1v), rankdata(gv) / len(gv); b = np.polyfit(rg, rs, 1); fit = np.polyval(b, rg)
    surr = spun_parcel_maps(rs - fit, SpinNull(axis, 1000, seed=20261008)); FIT = parcel_mean(fit); top = lambda v: np.argsort(-v)[:20]
    obs = M[:, top(GEO)].mean(1) - M[:, top(G1)].mean(1); nul = np.array([M[:, top(FIT + s)].mean(1) - M[:, top(G1)].mean(1) for s in surr])
    pa = (1 + (nul >= obs[None]).sum(0)) / (1 + len(nul))
    res = {t: {"position_G1": pG[i], "position_semantic": pS[i], "difference": d[i], "p_spin": p[i], "q_fdr": q[i], "apex_enrichment": obs[i], "apex_p": pa[i]} for i, t in enumerate(T)}
    json.dump({t: {k: float(v) for k, v in r.items()} for t, r in res.items()}, open(OUT / "extended_data" / "neurosynth_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
