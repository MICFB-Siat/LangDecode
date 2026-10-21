import sys, json, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "generalization"))
from scipy.stats import rankdata
from langdecoder.paths import OUT
from langdecoder.surface import parcel_mean
from langdecoder.spin import SpinNull, spun_parcel_maps, p_one_sided
from langdecoder.stats import holm
from fig4_gradient_correspondence import cifti_axis
from common import DIRECTIONS, g1_vertex, shared_map_vertex, load_region_2afc


def main():
    gv = shared_map_vertex(); geo, g1 = parcel_mean(gv), parcel_mean(g1_vertex()); y = np.nanmean([load_region_2afc(d) for d in DIRECTIONS], 0)
    ok = np.isfinite(geo) & np.isfinite(g1) & np.isfinite(y); y, n = y[ok], ok.sum(); t3 = np.minimum((rankdata(g1[ok]) / n * 3).astype(int), 2)
    def stat(g):
        return np.array([y[(t3 == q) & (g > np.median(g[t3 == q]))].mean() - y[(t3 == q) & (g <= np.median(g[t3 == q]))].mean() for q in range(3)])
    g = rankdata(geo[ok]); obs = stat(g)
    null = np.array([stat(rankdata(s[ok])) for s in spun_parcel_maps(gv, SpinNull(cifti_axis(), 1000, seed=20261008))])
    p = p_one_sided(obs, null); half = np.array([g[i] > np.median(g[t3 == t3[i]]) for i in range(n)])
    pd.DataFrame({"g1_third": np.array(["Low", "Mid", "High"])[t3], "half": np.where(half, "Higher", "Lower"), "single_region_2afc": y}).to_csv(OUT / "extended_data" / "ed3c_parcels.csv", index=False)
    json.dump({"higher_minus_lower": obs.tolist(), "spin_p": p.tolist(), "spin_p_holm": holm(p).tolist()}, open(OUT / "extended_data" / "ed3c_stats.json", "w"), indent=1); print(obs, p, holm(p))


if __name__ == "__main__":
    main()
