import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TRUNK_PREFIXES = ("blocks.", "pos", "tr.", "attn.")
SUBJECT_PREFIXES = ("spatial.", "film.")


class SubjectSpatial(nn.Module):
    def __init__(self, n_sub, k_vox, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_sub, k_vox, d) / math.sqrt(k_vox))
        self.b = nn.Parameter(torch.zeros(n_sub, d))

    def forward(self, x, sub):
        return x @ self.w[sub] + self.b[sub]


class FiLM(nn.Module):
    def __init__(self, n_sub, d):
        super().__init__()
        self.emb = nn.Embedding(n_sub, 2 * d)
        nn.init.zeros_(self.emb.weight)

    def forward(self, h, sub):
        g, b = self.emb.weight[sub].chunk(2, -1)
        return h * (1 + g) + b


class Net2(nn.Module):
    def __init__(self, n_sub, k_vox, n_delays=7, d=256, depth=3, n_head=4, n_layer=2, out=1536, drop=0.3):
        super().__init__()
        self.spatial = SubjectSpatial(n_sub, k_vox, d)
        self.film = FiLM(n_sub, d)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(d, d, 3, padding=2 ** i, dilation=2 ** i), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(drop))
            for i in range(depth)])
        self.pos = nn.Parameter(torch.zeros(1, n_delays, d))
        self.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, n_head, 4 * d, drop, batch_first=True, norm_first=True), n_layer)
        self.attn = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, out))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.gate = nn.Parameter(torch.ones(1), requires_grad=False)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def reset_head(self):

        nn.init.normal_(self.head[-1].weight, std=0.02); nn.init.zeros_(self.head[-1].bias)

    def pooled(self, x, sub):
        h = self.film(self.spatial(x, sub) + self.pos, sub).transpose(1, 2)
        for b in self.blocks:
            h = h + b(h)
        h = self.tr(h.transpose(1, 2))
        return (h * torch.softmax(self.attn(h), dim=1)).sum(1)

    def forward(self, x, sub):
        return self.gate * self.head(self.pooled(x, sub))


def loss_fn(pred, target, logit_scale):

    pn, yn = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
    logits = pn @ yn.T * logit_scale.exp().clamp(max=100.0)
    lab = torch.arange(len(pn), device=pn.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) + (1 - (pn * yn).sum(-1).mean())


def load_pretrained(net, path, device="cpu"):

    sd = torch.load(path, map_location=device, weights_only=False)
    sd = sd.get("state_dict", sd)
    own = net.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and not k.startswith(SUBJECT_PREFIXES)}
    net.load_state_dict(keep, strict=False)
    net.reset_head()
    return len(keep)


def trunk_state(net):

    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if k.startswith(TRUNK_PREFIXES)}
