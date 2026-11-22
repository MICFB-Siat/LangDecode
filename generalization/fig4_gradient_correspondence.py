import json, numpy as np, pandas as pd, nibabel as nib
from scipy.stats import rankdata
from langdecoder.paths import OUT, LPPC
from langdecoder.surface import parcel_mean
from langdecoder.spin import SpinNull, spun_parcel_maps, p_one_sided
from common import g1_vertex, shared_map_vertex, thirds, tier_vertex_labels, spearman


def cifti_axis():
    f = next(LPPC.glob("sub-*/*.dtseries.nii")); return nib.load(f).header.get_axis(1)


def main():
    g1v, gv = g1_vertex(), shared_map_vertex(); g1, geo = parcel_mean(g1v), parcel_mean(gv)
    rho = spearman(geo, g1); null = [spearman(geo, s) for s in spun_parcel_maps(g1v, SpinNull(cifti_axis(), 1000, seed=20261008))]
    dec = np.minimum((rankdata(g1) / 400 * 10).astype(int), 9); z = (rankdata(geo) - rankdata(geo).mean()) / rankdata(geo).std()
    pd.DataFrame({"G1_decile": np.arange(1, 11), "shared_map_mean": [geo[dec == k].mean() for k in range(10)],
                  "shared_map_z_rank_mean": [z[dec == k].mean() for k in range(10)], "n": [(dec == k).sum() for k in range(10)]}).to_csv(OUT / "generalization" / "gradient_correspondence.csv", index=False)
    json.dump({"rho": rho, "spin_p_one_sided": float(p_one_sided(rho, null)), "n_parcels": int(np.isfinite(geo).sum())}, open(OUT / "generalization" / "gradient_correspondence.json", "w"), indent=1)
    pct = lambda v: (rankdata(v) - 1) / (len(v) - 1)
    np.savez(OUT / "generalization" / "gradient_tiers_vertex.npz", g1_percentile=pct(g1v), shared_percentile=pct(gv),
             g1_tier=tier_vertex_labels(thirds(g1)), shared_tier=tier_vertex_labels(thirds(geo)))
    print(f"rho = {rho:.3f}")


if __name__ == "__main__":
    main()
