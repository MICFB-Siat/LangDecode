import argparse, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from langdecoder.model import Net2, load_pretrained
from common import NEB, MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, choices=["smn", "cn", "fr"]); ap.add_argument("--frac", type=float, required=True)
ap.add_argument("--keep", default="all"); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--epochs", type=int, default=10)
a = ap.parse_args()
dev, TR, ND = "cuda:0", 2.0, np.arange(0, 14, 2, dtype=np.float64); torch.manual_seed(0); np.random.seed(0)
src = "fr" if a.init == "fr" else "cn"
N = np.load(NEB / f"norm_{src}.npz"); fmean, fstd = (torch.tensor(N[k], device=dev) for k in ("fmean", "fstd"))
zz = np.load(NEB / f"zeroshot_{src}_pred.npz"); TK, TG = list(zz["targets_keys"]), zz["targets"].astype(np.float32)
tgt = lambda p: TG[TK.index(f"French|{p}")]
Z = pd.read_csv(NEB / "zeroshot_cn.csv"); Z = Z[(Z.keep == "all") & (Z.cond == "intact") & (Z.native == "French") & (Z.heard_lang == "French")]
subs = sorted(s for s in Z["sub"].unique() if (NEB / "ts" / f"{s}.npz").exists())
rng = np.random.default_rng(0); TRAIN, TEST = set(), set()
for par in (0, 1):
    ps = rng.permutation([p for p in range(1, 25) if p % 2 == par]); TRAIN |= set(int(p) for p in ps[:6]); TEST |= set(int(p) for p in ps[6:])
if a.fold == 1:
    TRAIN, TEST = TEST, TRAIN
train_subs = set(list(np.random.default_rng(100).permutation(subs))[:max(1, int(round(a.frac * len(subs))))])

RUNS, BLOCKS = {}, []
for s in subs:
    z = np.load(NEB / "ts" / f"{s}.npz")
    for r in (1, 2, 3):
        b = torch.tensor(z[f"run{r}"].astype(np.float32), device=dev); RUNS[(s, r)] = ((b - b.mean(0)) / b.std(0).clamp_min(1e-6)).half()
    ev = z["events"]
    BLOCKS += [(s, int(e["run"]), int(e["passage"]), float(e["onset"]), float(e["duration"])) for e in ev[(ev["cond"] == "intact") & (ev["lang"] == "French")]]
DROP = None
if a.keep != "all":
    km = np.zeros(59412, bool); km[np.load(NEB / "parcel_tertiles_geo.npz")[a.keep]] = True; DROP = torch.from_numpy(~km).to(dev)


def qpos(onset, dur, n):

    pos = (onset + np.arange(0, dur, 0.5))[:, None] / TR + ND[None] / TR; return pos[pos.max(1) < n - 1]


def make_x(items):
    xs = []
    for s, r, pos in items:
        run = RUNS[(s, r)]; lo = np.floor(pos).astype(np.int64); w = torch.tensor(pos - lo, dtype=torch.float32, device=dev)[:, None]
        lt = torch.from_numpy(lo).to(dev); xs.append(run[lt].float() + (run[lt + 1].float() - run[lt].float()) * w)
    x = (torch.stack(xs) - fmean) / fstd
    if DROP is not None:
        x[..., DROP] = 0.0
    return x


TRL = sorted(TRAIN); TI = {p: i for i, p in enumerate(TRL)}; YTR = torch.tensor(np.stack([tgt(p) for p in TRL]), device=dev)
SAMPLES = [(s, r, row, TI[p]) for s, r, p, on, du in BLOCKS if s in train_subs and p in TRAIN for row in qpos(on, du, RUNS[(s, r)].shape[0])]
net = Net2(1, int(N["k_vox"]), int(N["n_delay"]), int(N["d"]), int(N["depth"]), 4, int(N["layers"]), 1536, float(N["drop"])).to(dev)
if a.init == "smn":
    load_pretrained(net, MODELS / "trunk_smn.pt", dev)
else:
    sd = torch.load(MODELS / f"wc_{a.init}.pt", map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
    net.load_state_dict({k: (v.float().mean(0, keepdim=True) if k in ("spatial.w", "spatial.b", "film.emb.weight") else v) for k, v in sd.items()})


@torch.no_grad()
def evaluate():

    net.eval(); pred = {}
    for s in subs:
        bl = sorted([b for b in BLOCKS if b[0] == s and b[2] in TEST], key=lambda b: b[2])
        if len(bl) < 3:
            continue
        P = []
        for _, r, p, on, du in bl:
            x = make_x([(s, r, q) for q in qpos(on, du, RUNS[(s, r)].shape[0])])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.append(net(x, 0).float().mean(0).cpu().numpy())
        pred[s] = (np.array([b[2] for b in bl]), np.stack(P))
    net.train(); return pred


opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.1); net.train()
for ep in range(a.epochs):
    perm = np.random.permutation(len(SAMPLES))
    for i in range(0, len(perm), 128):
        bt = [SAMPLES[j] for j in perm[i:i + 128]]; yi = torch.tensor([t for *_, t in bt], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = net(make_x([(s, r, pos) for s, r, pos, _ in bt]), 0).float()
        pn, yn = F.normalize(pr, dim=-1), F.normalize(YTR, dim=-1)
        loss = F.cross_entropy(net.logit_scale.exp().clamp(max=100) * pn @ yn.T, yi) + (1 - (pn * yn[yi]).sum(-1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
pred = evaluate(); out = NEB / "fewshot_tier"; out.mkdir(parents=True, exist_ok=True)
tag = f"French_{a.keep}_{a.init}_f{a.frac:g}_fold{a.fold}"
np.savez_compressed(out / f"{tag}_pred.npz", **{f"{k}|p": v[0] for k, v in pred.items()}, **{f"{k}|x": v[1] for k, v in pred.items()})
json.dump(dict(vars(a), n_listeners=len(subs), n_train_listeners=len(train_subs), train_passages=sorted(TRAIN)), open(out / f"{tag}.json", "w"))
