import sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "generalization"))
from scipy.stats import spearmanr
from langdecoder.paths import OUT
from langdecoder.surface import parcel_mean
from common import PAIRS, DIRECTIONS, TRAIN_STORIES, load_region_2afc
from shared_map import ranks


def main(n_perm=1000):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = {L: torch.nan_to_num(torch.from_numpy(np.load(OUT / "semantic_geometry" / f"group_searchlight_rdm_{L}.npy", mmap_mode="r")[:TRAIN_STORIES].astype(np.float32))).to(dev) for L in ("CN", "EN", "FR")}
    RA = {L: ranks(G[L]) for L in G}; iu = np.triu_indices(24, 1); M = np.zeros((24, 24), int); M[iu] = np.arange(276); M = M + M.T
    fmap = lambda a, b: torch.tanh(torch.atanh((a * b).sum(-1).clamp(-0.999, 0.999)).mean(0)).cpu().numpy()
    y = np.nanmean([load_region_2afc(d) for d in DIRECTIONS], 0); ok = np.isfinite(y)
    real = np.mean([parcel_mean(fmap(RA[p[:2]], RA[p[3:]])) for p in PAIRS], 0); r_obs = spearmanr(real[ok], y[ok])[0]
    rng = np.random.default_rng(20260930); perms = np.stack([[rng.permutation(24) for _ in range(TRAIN_STORIES)] for _ in range(n_perm)]); null = []
    for p in range(n_perm):
        idx = torch.from_numpy(np.stack([M[perms[p, s][iu[0]], perms[p, s][iu[1]]] for s in range(TRAIN_STORIES)])).to(dev)
        fake = np.mean([parcel_mean(fmap(RA[q[:2]], ranks(torch.gather(G[q[3:]], 2, idx[:, None, :].expand(-1, G[q[3:]].shape[1], -1))))) for q in PAIRS], 0)
        null.append(spearmanr(fake[ok], y[ok])[0])
    null = np.array(null); pd.DataFrame({"null_rho": null}).to_csv(OUT / "extended_data" / "ed3b_null.csv", index=False)
    json.dump({"rho_matched": float(r_obs), "p_one_sided": float((1 + np.sum(null >= r_obs)) / (1 + n_perm)), "null_95": np.percentile(null, [2.5, 97.5]).tolist(), "n_parcels": int(ok.sum())},
              open(OUT / "extended_data" / "ed3b_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
