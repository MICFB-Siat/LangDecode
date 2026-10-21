import nibabel as nib
import numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon
from langdecoder.spin import SpinNull
from langdecoder.surface import parcel_mean
from langdecoder.stats import holm
from _common import OUT3, LANGS, PAIRS, cohort, run_files

O = OUT3 / "individual_correspondence"; N_ROT = 300


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y); return np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]


def main():
    O.mkdir(parents=True, exist_ok=True); c = cohort(); m = np.load(OUT3 / "fig3gi_maps.npz")
    ax = nib.load(str(run_files(c["EN"][0])[0])).header.get_axis(1); spin = SpinNull(ax, n_spins=1000, seed=20261004)
    rot = [idx for _, idx in zip(range(N_ROT), spin.rotations())]
    P = {L: np.load(OUT3 / "searchlight" / f"participant_semantic_z_{L}.npy").astype(float) for L in LANGS}
    targets = {f"g_{L}": P[L].mean(0) for L in LANGS}; targets["shared"] = np.mean([m[f"cross_{a}_{b}_rho"] for a, b in PAIRS], 0)
    tp = {k: parcel_mean(v) for k, v in targets.items()}
    tr = {k: [parcel_mean(spin.to_full(np.nan_to_num(v))[i]) for i in rot] for k, v in targets.items()}
    rows = []
    for L in LANGS:
        for sub, z in zip(c[L], P[L]):
            x = parcel_mean(z); r = {}
            for key, ks in (("h", [f"g_{M}" for M in LANGS if M != L]), ("j", ["shared"])):
                obs = np.mean([spearman(x, tp[k]) for k in ks]); null = np.mean([[spearman(x, q) for q in tr[k]] for k in ks], 0)
                r[f"{key}_rho"], r[f"{key}_null_median"] = obs, float(np.median(null))
            rows.append({"language": L, "participant": sub, **r})
    d = pd.DataFrame(rows); d.to_csv(O / "individual_correspondence.csv", index=False, float_format="%.5f"); st = []
    for key, name in (("h", "Fig3h"), ("j", "Fig3j")):
        pd.DataFrame({L: pd.Series(d[d.language == L][f"{key}_rho"].to_numpy()) for L in LANGS}).to_csv(O / f"{name}_prism.csv", index=False, float_format="%.5f")
        p = [wilcoxon(d[d.language == L][f"{key}_rho"] - d[d.language == L][f"{key}_null_median"], alternative="greater").pvalue for L in LANGS]
        st += [{"panel": name, "language": L, "p": pp, "p_holm": ph} for L, pp, ph in zip(LANGS, p, holm(p))]
    pd.DataFrame(st).to_csv(O / "fig3hj_stats.csv", index=False); print(pd.DataFrame(st))


if __name__ == "__main__":
    main()
