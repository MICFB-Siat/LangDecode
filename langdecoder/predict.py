import numpy as np
import torch

from .data import DATASETS, fir_index, load_run


def zscore(b):
    return (b - b.mean(0)) / np.maximum(b.std(0), 1e-6)


def group_feed(dataset, subjects, story, vox):

    parts = [zscore(load_run(dataset, s, story, vox)) for s in subjects]
    n = min(len(p) for p in parts)
    return zscore(np.mean([p[:n] for p in parts], 0)).astype(np.float32)


@torch.no_grad()
def trajectory(net, ck, feeds, times, tr, keep=None, device="cpu"):

    n_rows = min(len(v) for v in feeds.values())
    lo, hi, w, kept = fir_index(n_rows, times, ck["delays"], tr)
    lo, hi, w = (torch.from_numpy(a).to(device) for a in (lo, hi, w))
    fmean, fstd = ck["fmean"].to(device), ck["fstd"].to(device)
    drop = None if keep is None else torch.from_numpy(~np.asarray(keep, bool)).to(device)
    acc = torch.zeros(len(lo), 1536, device=device)
    for i, bx in feeds.items():
        bt = torch.from_numpy(np.asarray(bx[:n_rows], np.float32)).to(device)
        for s in range(0, len(lo), 1024):
            sl = slice(s, min(s + 1024, len(lo)))
            a, c = bt[lo[sl]], bt[hi[sl]]
            x = ((a + (c - a) * w[sl].unsqueeze(-1)) - fmean) / fstd
            if drop is not None:
                x[..., drop] = 0.0
            acc[sl] += net(x, i)
    return (acc / len(feeds)).cpu().numpy(), kept


def feeds_for(ck, dataset, story, mode, target_subjects=None):

    vox = ck["vox"]
    if mode == "own":
        return {i: zscore(load_run(dataset, s, story, vox)) for i, s in enumerate(ck["subjects"])}
    g = group_feed(dataset, target_subjects, story, vox)
    return {i: g for i in range(len(ck["subjects"]))}


def tr_of(dataset):
    return DATASETS[dataset]["tr"]
