import numpy as np, pandas as pd, torch
from langdecoder.paths import OUT
from langdecoder.surface import parcel_mean
from common import PAIRS, TRAIN_STORIES, thirds, shared_map_vertex


def ranks(x):
    r = x.argsort(-1).argsort(-1).float(); r = r - r.mean(-1, keepdim=True); return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; (OUT / "generalization").mkdir(parents=True, exist_ok=True)
    R = {L: ranks(torch.nan_to_num(torch.from_numpy(np.load(OUT / "semantic_geometry" / f"group_searchlight_rdm_{L}.npy", mmap_mode="r")[:TRAIN_STORIES].astype(np.float32))).to(dev))
         for L in ("CN", "EN", "FR")}
    for pair in PAIRS:
        a, b = pair.split("_"); rho = (R[a] * R[b]).sum(-1)
        np.save(OUT / "generalization" / f"shared_map_train_{pair}.npy", torch.tanh(torch.atanh(rho.clamp(-0.999, 0.999)).mean(0)).cpu().numpy())
    gp = parcel_mean(shared_map_vertex())
    pd.DataFrame({"parcel": np.arange(1, 401), "shared_map": gp, "tier": np.array(["Low", "Mid", "High"])[thirds(gp)]}).to_csv(OUT / "generalization" / "shared_map_parcels.csv", index=False)


if __name__ == "__main__":
    main()
