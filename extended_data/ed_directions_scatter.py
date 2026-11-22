import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "generalization"))
from scipy.stats import spearmanr
from langdecoder.paths import OUT
from langdecoder.surface import parcel_mean
from langdecoder.spin import SpinNull
from langdecoder.stats import holm
from fig4_gradient_correspondence import cifti_axis
from common import DIRECTIONS, pair_of, shared_map_vertex, load_region_2afc


def main():
    sp = SpinNull(cifti_axis(), 1000, seed=404); rot = list(sp.rotations()); rows, pts = [], []
    for d in DIRECTIONS:
        gv = shared_map_vertex(pair_of(d)); x, y = parcel_mean(gv), load_region_2afc(d); ok = np.isfinite(x) & np.isfinite(y)
        r = spearmanr(x[ok], y[ok])[0]; full = sp.to_full(gv); nl = []
        for idx in rot:
            xr = parcel_mean(full[idx]); m = ok & np.isfinite(xr); nl.append(spearmanr(xr[m], y[m])[0])
        rows.append({"direction": d, "rho": r, "spin_p_one_sided": (1 + np.sum(np.array(nl) >= r)) / (1 + len(nl)), "n_parcels": int(ok.sum())})
        pts.append(pd.DataFrame({"direction": d, "parcel": np.flatnonzero(ok) + 1, "semantic_geometry": x[ok], "single_region_2afc": y[ok]}))
    S = pd.DataFrame(rows); S["spin_p_holm"] = holm(S.spin_p_one_sided.to_numpy())
    S.to_csv(OUT / "extended_data" / "ed5_stats.csv", index=False); pd.concat(pts).to_csv(OUT / "extended_data" / "ed5_points.csv", index=False); print(S)


if __name__ == "__main__":
    main()
