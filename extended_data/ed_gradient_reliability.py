import sys, json, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "generalization"))
from scipy.stats import rankdata, spearmanr
from langdecoder.paths import OUT, LANGS
from langdecoder.surface import parcel_labels, parcel_mean
from langdecoder.spin import SpinNull
from fig4_gradient_correspondence import cifti_axis
from common import g1_vertex, shared_map_vertex


def reliability():
    par = parcel_labels(); REL = np.full((3, 400), np.nan)
    for li, L in enumerate(LANGS):
        z = np.load(OUT / "semantic_geometry" / f"group_halves_{L}.npz"); H, valid = z["half"].astype(np.float32), z["valid"].astype(bool); iu = np.triu_indices(H.shape[1], 1)
        for k in range(1, 401):
            m = (par == k) & valid
            if m.sum() < 10: continue
            R = []
            for h in (0, 1):
                X = H[h][:, m]; X = (X - X.mean(0)) / (X.std(0) + 1e-6); X -= X.mean(1, keepdims=True); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9; R.append((1 - X @ X.T)[iu])
            REL[li, k - 1] = spearmanr(R[0], R[1])[0]
    return REL


def resid(y, C):
    X = np.column_stack([np.ones(len(y))] + [rankdata(c) for c in C]); r = rankdata(y); return r - X @ np.linalg.lstsq(X, r, rcond=None)[0]


def main():
    REL = reliability(); g1v = g1_vertex(); geo, g1 = parcel_mean(shared_map_vertex()), parcel_mean(g1v)
    ok = np.isfinite(geo) & np.isfinite(g1) & np.isfinite(REL).all(0); C = [r[ok] for r in REL]
    rho = spearmanr(geo[ok], g1[ok])[0]; part = np.corrcoef(resid(geo[ok], C), resid(g1[ok], C))[0, 1]
    sp = SpinNull(cifti_axis(), 1000, seed=20261006); full = sp.to_full(g1v); nl = []
    for idx in sp.rotations():
        g = parcel_mean(full[idx])[ok]; m = np.isfinite(g); nl.append(np.corrcoef(resid(geo[ok][m], [c[m] for c in C]), resid(g[m], [c[m] for c in C]))[0, 1])
    p = (1 + np.sum(np.array(nl) >= part)) / (1 + len(nl))
    zr = lambda v: (v - v.mean()) / v.std(); raw, adj = zr(rankdata(geo[ok])), zr(resid(geo[ok], C)); dec = np.minimum((rankdata(g1[ok]) / ok.sum() * 10).astype(int), 9)
    pd.DataFrame({"G1_decile": np.arange(1, 11), **{f"{n}_{s}": [f(v[dec == k]) for k in range(10)] for n, v in (("raw", raw), ("reliability_adjusted", adj))
                  for s, f in (("mean", np.mean), ("sem", lambda x: x.std(ddof=1) / np.sqrt(len(x))))}, "n": [(dec == k).sum() for k in range(10)]}).to_csv(OUT / "extended_data" / "ed3a_profile.csv", index=False)
    json.dump({"rho": float(rho), "partial_rho": float(part), "spin_p_partial_one_sided": float(p), "n_parcels": int(ok.sum())}, open(OUT / "extended_data" / "ed3a_stats.json", "w"), indent=1)
    np.save(OUT / "extended_data" / "parcel_reliability.npy", REL); print(rho, part, p)


if __name__ == "__main__":
    main()
