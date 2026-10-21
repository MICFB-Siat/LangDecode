import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from langdecoder.paths import N_CORTEX
from .data import DELAYS, Store, feature_samples, feature_stats, load_tokens, subjects
from .model import Net2, load_pretrained, loss_fn, trunk_state


def story_names(spec, fmt="story{:02d}"):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += [fmt.format(i) for i in range(int(a), int(b or a) + 1)]
    return out


def target_stats(pairs, stories, smooth):
    y = np.concatenate([load_tokens(ds, lang, st, smooth)[1] for ds, lang in pairs for st in stories])
    return y.mean(0).astype(np.float32), np.maximum(y.std(0), 1e-6).astype(np.float32)


def vertices(vox_file):
    return np.sort(np.load(vox_file).astype(np.int64)) if vox_file else np.arange(N_CORTEX)


def train_step(net, opt, sched, x, y, sub, noise):
    if noise:
        x = x + noise * torch.randn_like(x)
    loss = loss_fn(net(x, sub), y, net.logit_scale)
    opt.zero_grad(set_to_none=True); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return float(loss)


def group_prediction(net, store, story, fmean, fstd, sub_ids=None, mask=None):

    net.eval(); n = store.tok[story]["n"]; dev = store.device
    sub_ids = range(len(store.subjects)) if sub_ids is None else sub_ids; out = torch.zeros(n, net.head[-1].out_features, device=dev)
    with torch.no_grad():
        for i in sub_ids:
            for s in range(0, n, 1024):
                idx = torch.arange(s, min(s + 1024, n), device=dev)
                x = (store.features(story, idx, torch.full((len(idx),), i, device=dev, dtype=torch.long)) - fmean) / fstd
                if mask is not None:
                    x[..., mask] = 0.0
                out[idx] += net(x, i)
    return out / len(sub_ids)


def pretrain(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(a.device)
    stories = story_names(a.stories); fit = [s for s in stories if s != a.val_story]
    vox, srcs = vertices(a.vox_file), [tuple(s.split(":")) for s in a.sources.split(",")]
    stores, offs, base = {}, {}, 0
    for ds, lang in srcs:
        stores[(ds, lang)] = Store(ds, lang, subjects(ds, lang)[:a.max_subjects], stories, vox, dev, DELAYS, a.smooth)
        offs[(ds, lang)] = base; base += len(stores[(ds, lang)].subjects)
    ym, ys = (torch.from_numpy(v).to(dev) for v in target_stats(srcs, fit, a.smooth))
    fmean, fstd = feature_stats(torch.cat([feature_samples(stores[k], fit[:4], 300) for k in srcs]))
    net = Net2(base, len(vox), len(DELAYS), a.d, a.depth, 4, a.layers, 1536, a.drop).to(dev); net.reset_head()
    if a.init:
        load_pretrained(net, a.init, dev)
    net.gate.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = max(1, sum(len(stores[k].subjects) * sum(stores[k].tok[st]["n"] for st in fit) for k in srcs) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=0.15)
    for ep in range(a.epochs):
        net.train(); chunks = []
        for k in srcs:
            for st in fit:
                for i in range(len(stores[k].subjects)):
                    p = np.random.permutation(stores[k].tok[st]["n"])
                    chunks += [(k, st, i, p[s:s + a.batch]) for s in range(0, len(p) - a.batch + 1, a.batch)]
        np.random.shuffle(chunks); tot = 0.0
        for k, st, i, idxn in chunks:
            idx = torch.from_numpy(idxn).to(dev); S = stores[k]
            x = (S.features(st, idx, torch.full((len(idx),), i, device=dev, dtype=torch.long)) - fmean) / fstd
            tot += train_step(net, opt, sched, x, (S.tok[st]["y"][idx] - ym) / ys, offs[k] + i, a.noise)
        val = {}
        for k in srcs:
            S = stores[k]
            p = group_prediction(_Offset(net, offs[k]), S, a.val_story, fmean, fstd); y = (S.tok[a.val_story]["y"] - ym) / ys
            val[":".join(k)] = round(float(F.cosine_similarity(p, y, dim=-1).mean()), 5)
        print(json.dumps({"epoch": ep, "loss": round(tot / max(len(chunks), 1), 4), "val_cosine": val}), flush=True)
        torch.save(trunk_state(net), a.out)


class _Offset(nn.Module):


    def __init__(self, net, off):
        super().__init__(); self.net, self.off, self.head = net, off, net.head

    def forward(self, x, i):
        return self.net(x, self.off + i)


def finetune(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(a.device)
    fmt = "story{:02d}" if a.dataset != "smn4lang" else "run{:02d}"
    fit, val, test = story_names(a.fit, fmt), story_names(a.val, fmt), story_names(a.test, fmt)
    vox = vertices(a.vox_file); subs = a.subjects.split(",") if a.subjects else subjects(a.dataset, a.lang)
    S = Store(a.dataset, a.lang, subs, fit + val + test, vox, dev, DELAYS, a.smooth)
    ym, ys = (torch.from_numpy(v).to(dev) for v in target_stats([(a.dataset, a.lang)], fit, a.smooth))
    fmean, fstd = feature_stats(feature_samples(S, fit[:12], 400))
    net = Net2(len(subs), len(vox), len(DELAYS), a.d, a.depth, 4, a.layers, 1536, a.drop).to(dev)
    if a.init:
        load_pretrained(net, a.init, dev)
    else:
        net.reset_head()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    steps = max(1, sum(S.tok[r]["n"] for r in fit) * len(subs) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=0.15)

    def score(stories):
        p = torch.cat([group_prediction(net, S, r, fmean, fstd) for r in stories])
        y = torch.cat([(S.tok[r]["y"] - ym) / ys for r in stories])
        return float(F.cosine_similarity(p, y, dim=-1).mean())

    best = None
    for ep in range(a.epochs):
        net.train(); chunks = []
        for r in fit:
            for i in range(len(subs)):
                p = np.random.permutation(S.tok[r]["n"])
                chunks += [(r, i, p[s:s + a.batch]) for s in range(0, len(p) - a.batch + 1, a.batch)]
        np.random.shuffle(chunks); tot = 0.0
        for r, i, idxn in chunks:
            idx = torch.from_numpy(idxn).to(dev)
            x = (S.features(r, idx, torch.full((len(idx),), i, device=dev, dtype=torch.long)) - fmean) / fstd
            tot += train_step(net, opt, sched, x, (S.tok[r]["y"][idx] - ym) / ys, i, a.noise)
        v = score(val); print(json.dumps({"epoch": ep, "loss": round(tot / max(len(chunks), 1), 4), "val_cosine": round(v, 5)}), flush=True)
        if best is None or v > best[0]:
            best = (v, ep, {k: t.detach().clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[2])
    print(json.dumps({"best_epoch": best[1], "test_cosine": round(score(test), 5)}), flush=True)
    torch.save({"state_dict": net.state_dict(), "fmean": fmean.cpu(), "fstd": fstd.cpu(), "ymean": ym.cpu(), "ystd": ys.cpu(),
                "subjects": subs, "vox": vox, "delays": DELAYS, "smooth": a.smooth, "dataset": a.dataset, "lang": a.lang,
                "arch": {"d": a.d, "depth": a.depth, "layers": a.layers, "drop": a.drop}}, a.out)


def load_decoder(path, device="cpu"):

    ck = torch.load(path, map_location=device, weights_only=False); ar = ck["arch"]
    net = Net2(len(ck["subjects"]), len(ck["vox"]), len(ck["delays"]), ar["d"], ar["depth"], 4, ar["layers"], 1536, ar["drop"]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pretrain", "finetune"):
        p = sp.add_parser(name)
        p.add_argument("--out", required=True); p.add_argument("--vox-file", default="", help="npy of cortical vertices (default: whole cortex)")
        p.add_argument("--device", default="cuda:0"); p.add_argument("--seed", type=int, default=0)
        p.add_argument("--d", type=int, default=256); p.add_argument("--depth", type=int, default=3); p.add_argument("--layers", type=int, default=2)
        p.add_argument("--drop", type=float, default=0.3); p.add_argument("--wd", type=float, default=0.1)
    p = sp.choices["pretrain"]
    p.add_argument("--sources", default="lppc:cn,lppc:en,lppc:fr,chen:en"); p.add_argument("--max-subjects", type=int, default=40)
    p.add_argument("--stories", default="1-9"); p.add_argument("--val-story", default="story09")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch", type=int, default=512); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--noise", type=float, default=0.1); p.add_argument("--smooth", type=int, default=4); p.add_argument("--init", default="")
    p = sp.choices["finetune"]
    p.add_argument("--dataset", default="lppc"); p.add_argument("--lang", default="en"); p.add_argument("--subjects", default="")
    p.add_argument("--fit", default="1-7"); p.add_argument("--val", default="8"); p.add_argument("--test", default="9")
    p.add_argument("--init", default=""); p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--noise", type=float, default=0.05); p.add_argument("--smooth", type=int, default=5)
    a = ap.parse_args()
    pretrain(a) if a.cmd == "pretrain" else finetune(a)


if __name__ == "__main__":
    main()
