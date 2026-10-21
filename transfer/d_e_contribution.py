import json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from langdecoder.surface import parcel_mean
from langdecoder.spin import SpinNull, spun_parcel_maps, p_one_sided
from langdecoder.stats import holm, stars
from common import F5, RSA, cifti_axis, thirds

Z = []
for T in ("cn", "en", "fr"):
    z = np.load(F5 / f"contrib_lppc_{T}_p30_xl.npz"); v = np.full(400, np.nan); v[z["parcels"].astype(int) - 1] = z["drop_cos"]
    Z.append((v - np.nanmean(v)) / np.nanstd(v))
Z = np.nanmean(Z, 0); m = np.isfinite(Z); P = np.full(400, np.nan); P[m] = (rankdata(Z[m]) - 0.5) / m.sum()

gv = (np.load(RSA / "xrsa_train_CN_EN.npy") + np.load(RSA / "xrsa_train_EN_FR.npy")) / 2
gv = np.where(np.isfinite(gv), gv, np.nanmean(gv)); G = parcel_mean(gv)


def stat(g):
    t = thirds(g); ok = np.isfinite(P) & np.isfinite(g)
    return np.array([P[ok & (t == 1)].mean() - P[ok & (t == 0)].mean(), P[ok & (t == 2)].mean() - P[ok & (t == 1)].mean()])


obs = stat(G); null = np.array([stat(s) for s in spun_parcel_maps(gv, SpinNull(cifti_axis(), 1000, seed=20261008))])
p = p_one_sided(obs, null); q = holm(p)
res = {k: dict(diff=float(d), spin_p=float(a), holm_p=float(b), stars=stars(b)) for k, d, a, b in zip(("Mid-Low", "High-Mid"), obs, p, q)}
json.dump(res, open(F5 / "fig5e_spin_tests.json", "w"), indent=1); print(json.dumps(res, indent=1))
t = thirds(G); pd.DataFrame({lab: pd.Series(P[t == j]) for j, lab in enumerate(("Low", "Mid", "High"))}).to_csv(
    F5 / "prism_fig5e_contribution_by_group.csv", index=False, float_format="%.4f")
np.save(F5 / "fig5d_contribution_percentile_parcels.npy", P)
