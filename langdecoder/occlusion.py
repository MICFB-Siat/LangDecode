import numpy as np
import torch
import torch.nn.functional as F

from .data import load_tokens
from .predict import trajectory


def region_mask(vox, labels, ids):

    return np.isin(np.asarray(labels)[np.asarray(vox)], np.atleast_1d(ids))


def targets(ck, dataset, lang, story, kept):
    y = load_tokens(dataset, lang, story, ck["smooth"])[1][kept]
    return (y - ck["ymean"].numpy()) / ck["ystd"].numpy()


def cosine(pred, y):
    return float(F.cosine_similarity(torch.from_numpy(pred), torch.from_numpy(np.asarray(y, np.float32)), dim=-1).mean())


def segment_2afc(pred, y, times, seg_s=20.0):
    seg = ((times - times[0]) // seg_s).astype(int); ns = seg.max() + 1

    def seg_mean(a):
        a = F.normalize(torch.from_numpy(np.asarray(a, np.float32)), dim=-1); m = torch.zeros(ns, a.shape[1])
        m.index_add_(0, torch.from_numpy(seg), a); return F.normalize(m, dim=-1)
    S = seg_mean(pred) @ seg_mean(y).T; d = S.diagonal()
    c = (d[:, None] + d[None, :]) > (S + S.T); iu = torch.triu_indices(ns, ns, 1)
    return float(c[iu[0], iu[1]].float().mean())


def keep_only_sweep(net, ck, feeds, times, tr, y, labels, ids=None, min_vertices=10, device="cpu"):

    lab = np.asarray(labels)[ck["vox"]]; ids = np.array(sorted(set(lab.tolist()) - {0})) if ids is None else np.asarray(ids)
    cos, afc, nv = [], [], []
    for q in ids:
        k = lab == q; nv.append(int(k.sum()))
        if k.sum() < min_vertices:
            cos.append(np.nan); afc.append(np.nan); continue
        p, kept = trajectory(net, ck, feeds, times, tr, keep=k, device=device)
        cos.append(cosine(p, y)); afc.append(segment_2afc(p, y, times[kept]))
    return {"ids": ids, "cosine": np.array(cos), "two_afc": np.array(afc), "n_vertices": np.array(nv)}


def occlusion_contributions(net, ck, feeds, times, tr, y, masks, device="cpu"):

    base, _ = trajectory(net, ck, feeds, times, tr, device=device); b = cosine(base, y)
    return b, {k: b - cosine(trajectory(net, ck, feeds, times, tr, keep=~m, device=device)[0], y) for k, m in masks.items()}
