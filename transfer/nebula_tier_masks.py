import numpy as np
from langdecoder.surface import parcel_labels, parcel_mean
from common import RSA, NEB

parcel = parcel_labels()
geo = np.mean([np.load(RSA / f"xrsa_all_{p}.npy") for p in ("CN_EN", "CN_FR", "EN_FR")], 0)
P = parcel_mean(geo); ok = np.flatnonzero(np.isfinite(P)); parts = np.array_split(ok[np.argsort(P[ok], kind="stable")], 3)
rng = np.random.default_rng(20261005); out = {}
for k, ps in enumerate(parts):
    out[f"T{k + 1}"] = np.flatnonzero(np.isin(parcel, ps + 1))
for k, ps in enumerate(parts):
    out[f"R{k + 1}"] = np.flatnonzero(np.isin(parcel, rng.choice(ok, len(ps), replace=False) + 1))
np.savez(NEB / "parcel_tertiles_geo.npz", **out); print({k: len(v) for k, v in out.items()})
