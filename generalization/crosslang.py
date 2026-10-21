import numpy as np, torch, torch.nn.functional as F
from langdecoder.paths import OUT
from langdecoder.model import Net2

TR, DELAYS = 2.0, np.arange(0, 14, 2, dtype=float)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_decoder(name, untrained_seed=None):
    ck = torch.load(OUT / "decoders" / f"{name}.pt", map_location="cpu", weights_only=False); ar = ck["arch"]
    if untrained_seed is not None: torch.manual_seed(untrained_seed)
    net = Net2(len(ck["subjects"]), len(ck["vox"]), len(ck["delays"]), ar["d"], ar["depth"], 4, ar["layers"], 1536, ar["drop"])
    if untrained_seed is None: net.load_state_dict(ck["state_dict"])
    norm = {k: np.asarray(ck[k]) for k in ("fmean", "fstd", "ymean", "ystd")}
    norm.update(smooth=ck["smooth"], n_sub=len(ck["subjects"]), k_vox=len(ck["vox"]), n_delay=len(ck["delays"]), vox=np.asarray(ck["vox"]))
    return net.to(DEV).eval(), norm


def zscore(x):
    x = np.asarray(x, np.float32); return (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)


def group_feed(subjects, story="story09"):
    xs = [zscore(np.load(OUT / "bold" / s / f"bold_{story}.npy", mmap_mode="r")) for s in subjects]; n = min(len(x) for x in xs)
    return zscore(np.mean([x[:n] for x in xs], 0))


def delay_index(n_rows, times):
    pos = (np.asarray(times, float)[:, None] + DELAYS[None]) / TR; lo = np.floor(pos).astype(np.int64)
    keep = (lo >= 0).all(1) & (lo + 1 < n_rows).all(1); return lo[keep], (pos - np.floor(pos))[keep].astype(np.float32), keep


def causal_smooth(y, width):
    c = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(y, 0)]); i = np.arange(len(y)); lo = np.maximum(0, i + 1 - width)
    return ((c[i + 1] - c[lo]) / (i + 1 - lo)[:, None]).astype(np.float32)


def targets(lang, norm, story="story09"):
    z = np.load(OUT / "tokens" / lang / f"{story}.npz"); y = causal_smooth(z["targets"].astype(np.float32), int(norm["smooth"]))
    return z["times"].astype(float), (y - norm["ymean"]) / norm["ystd"]


@torch.no_grad()
def predict(net, norm, feeds, lo, w, keep_vertices=None, pathways=None):

    pathways = range(int(norm["n_sub"])) if pathways is None else pathways
    fm, fs = (torch.as_tensor(norm[k], device=DEV) for k in ("fmean", "fstd")); lo_t, w_t = torch.as_tensor(lo, device=DEV), torch.as_tensor(w, device=DEV)[..., None]
    zero = None if keep_vertices is None else torch.as_tensor(~keep_vertices, device=DEV)
    out = 0
    for i in pathways:
        b = torch.as_tensor(feeds[i] if isinstance(feeds, dict) else feeds, device=DEV); acc = []
        for s in range(0, len(lo), 1024):
            a = b[lo_t[s:s + 1024]]; x = (a + (b[lo_t[s:s + 1024] + 1] - a) * w_t[s:s + 1024] - fm) / fs
            if zero is not None: x[..., zero] = 0.0
            acc.append(net(x, i))
        out = out + torch.cat(acc)
    return out / len(pathways)


def two_afc(pred, y, times, seg_seconds=20):

    seg = torch.as_tensor(((times - times[0]) // seg_seconds).astype(np.int64), device=DEV); ns = int(seg.max()) + 1
    def mean(a): return F.normalize(torch.zeros(ns, a.shape[1], device=DEV).index_add_(0, seg, F.normalize(a, dim=-1)), dim=-1)
    S = mean(pred) @ mean(torch.as_tensor(y, device=DEV)).T; d = S.diagonal(); iu = torch.triu_indices(ns, ns, 1, device=DEV)
    return float(((d[:, None] + d[None]) > (S + S.T))[iu[0], iu[1]].float().mean())


def cosine(pred, y):
    return float(F.cosine_similarity(pred, torch.as_tensor(y, device=DEV), dim=-1).mean())
