import sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "generalization"))
from scipy.stats import spearmanr
from langdecoder.paths import DATA, OUT
from langdecoder.surface import parcel_labels, parcel_mean
from langdecoder.spin import SpinNull
from fig4_gradient_correspondence import cifti_axis
from common import shared_map_vertex


def main():
    par = parcel_labels(); geo = parcel_mean(shared_map_vertex())
    G = {f"G{k}": parcel_mean(np.load(DATA / "atlas" / f"margulies2016_g{k}_fslr32k.npy")) for k in (1, 2, 3)}
    sp = SpinNull(cifti_axis(), 1000, seed=20261010); full = sp.to_full(geo[np.clip(par - 1, 0, 399)]); null = [parcel_mean(full[i]) for i in sp.rotations()]
    out = {}
    for k, g in G.items():
        r = spearmanr(geo, g)[0]; nl = np.abs([spearmanr(x, g, nan_policy="omit")[0] for x in null])
        out[k] = {"rho": float(r), "spin_p_two_sided": float((1 + np.sum(nl >= abs(r))) / (1 + len(nl))), "null_abs": nl}
    for k in ("G2", "G3"):
        d = abs(out["G1"]["rho"]) - abs(out[k]["rho"]); dn = out["G1"]["null_abs"] - out[k]["null_abs"]
        out[k]["G1_minus_this_abs_rho"] = float(d); out[k]["p_one_sided"] = float((1 + np.sum(dn >= d)) / (1 + len(dn)))
    res = {k: {a: b for a, b in v.items() if a != "null_abs"} for k, v in out.items()}
    (OUT / "extended_data").mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "extended_data" / "ed_three_gradients.json", "w"), indent=1); print(res)


if __name__ == "__main__":
    main()
