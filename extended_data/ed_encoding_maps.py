import sys, itertools, numpy as np, nibabel as nib, torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from langdecoder.paths import OUT, LANGS, SPHERES, ATLAS, N_CORTEX


def tstat(z):
    return z.mean(0) / (z.std(0, ddof=1) / np.sqrt(len(z)) + 1e-12)


def maxT_fwe(z, n_perm=10000, seed=20260830):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; Z = torch.as_tensor(z, dtype=torch.float32, device=dev); n = len(z); obs = torch.as_tensor(tstat(z), device=dev)
    signs = torch.as_tensor(np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_perm, n)), dtype=torch.float32, device=dev); sq = (Z * Z).sum(0); mx = []
    for s in range(0, n_perm, 128):
        m = signs[s:s + 128] @ Z / n; mx.append((m / torch.sqrt(torch.clamp((sq - n * m * m) / (n - 1), min=1e-12) / n)).max(1).values)
    mx = torch.sort(torch.cat(mx)).values; exceed = n_perm - torch.searchsorted(mx, obs); return ((exceed + 1) / (n_perm + 1)).cpu().numpy()


def mesh_edges():
    idx = np.load(ATLAS, allow_pickle=True)["idx64984"]; inv = np.full(64984, -1); inv[idx] = np.arange(N_CORTEX); E = []
    for h, off in (("L", 0), ("R", 32492)):
        T = nib.load(str(SPHERES / f"tpl-fsLR_den-32k_hemi-{h}_midthickness.surf.gii")).darrays[1].data.astype(int) + off
        for a, b in ((0, 1), (1, 2), (0, 2)): e = np.stack([inv[T[:, a]], inv[T[:, b]]], 1); E.append(e[(e >= 0).all(1)])
    return np.concatenate(E)


def cluster_fwe(z, E, thr=2.015):
    def masses(zs):
        t = tstat(zs); k = np.flatnonzero(t > thr); pos = np.full(N_CORTEX, -1); pos[k] = np.arange(len(k)); e = E[(t[E[:, 0]] > thr) & (t[E[:, 1]] > thr)]
        n, lab = connected_components(sparse.coo_matrix((np.ones(len(e)), (pos[e[:, 0]], pos[e[:, 1]])), shape=(len(k), len(k))), directed=False)
        return (np.bincount(lab, weights=t[k], minlength=n) if n else np.zeros(1)), k, lab
    obs, k, lab = masses(z); flips = np.array(list(itertools.product([1, -1], repeat=len(z))), float)
    null = np.array([masses(z * f[:, None])[0].max() for f in flips]); p = np.array([(null >= m).sum() / len(flips) for m in obs])
    sig = np.zeros(N_CORTEX, bool)
    for c in np.flatnonzero(p < 0.05): sig[k[lab == c]] = True
    return sig


def main():
    maps = {}; i = 4
    for A in LANGS:
        subs = open(OUT / f"subjects_{A.lower()}.txt").read().split()
        for B in LANGS:
            z = np.stack([np.load(OUT / "encoding" / A / f"{s}.npz")[B] for s in subs]); sig = maxT_fwe(z, seed=20260830 + i) < 0.05; i += 1
            maps[f"LPPC_{A}_to_{B}"] = np.where(sig, np.tanh(z.mean(0)), np.nan)
    subs = open(OUT / "subjects_chen.txt").read().split(); E = mesh_edges()
    for A, B in (("en", "en"), ("zh", "zh"), ("en", "zh"), ("zh", "en")):
        z = np.nan_to_num(np.stack([np.load(OUT / "encoding_chen" / A / f"{s}.npz")[B] for s in subs]))
        maps[f"Chen2026_{A}_to_{B}"] = np.where(cluster_fwe(z, E), np.tanh(z.mean(0)), np.nan)
    np.savez(OUT / "extended_data" / "ed4_maps.npz", **maps); print({k: int(np.isfinite(v).sum()) for k, v in maps.items()})


if __name__ == "__main__":
    main()
