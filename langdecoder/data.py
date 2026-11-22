import re
from pathlib import Path

import numpy as np
import torch

from langdecoder.paths import DATA, N_CORTEX

FEATURE_LAYER, CONTEXT_TOKENS = 20, 15
DELAYS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)

DATASETS = {"lppc": {"tr": 2.0, "band": None}, "chen": {"tr": 2.0, "band": None}, "smn4lang": {"tr": 0.71, "band": (0.009, 0.20)}}
SILENCE = {"", "sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", "sil", "#"}


def textgrid_words(path, prefer=("word", "words", "TokensAlign")):

    raw = Path(path).read_bytes()
    content = raw.decode("utf-16") if raw[:2] in (b"\xfe\xff", b"\xff\xfe") else raw.decode("utf-8", errors="replace")
    tiers = []
    if "chronological" in content[:80]:
        names = re.findall(r'^"IntervalTier"\s+"([^"]*)"', content, flags=re.MULTILINE)
        per = {i + 1: [] for i in range(len(names))}
        for k, b, e, t in re.findall(r'^(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\n"((?:[^"]|"")*)"', content, flags=re.MULTILINE):
            per[int(k)].append((b, e, t))
        tiers = [(names[i], per[i + 1]) for i in range(len(names))]
    elif "xmin =" not in content[:400]:
        lines, i = [ln.strip() for ln in content.splitlines()], 0
        while i < len(lines):
            if lines[i] == '"IntervalTier"':
                name, n, j, ints = lines[i + 1].strip('"'), int(lines[i + 4]), i + 5, []
                for _ in range(n):
                    ints.append((lines[j], lines[j + 1], lines[j + 2].strip('"'))); j += 3
                tiers.append((name, ints)); i = j
            else:
                i += 1
    else:
        pat = re.compile(r'xmin = ([0-9.eE+-]+)[ \t]*\n\s*xmax = ([0-9.eE+-]+)[ \t]*\n\s*text = "(.*?)"[ \t]*(?=\n\s*intervals \[|\s*\Z)', re.DOTALL)
        for it in re.split(r"\n\s*item \[\d+\]:", content)[1:]:
            m = re.search(r'name = "([^"]*)"', it)
            tiers.append((m.group(1) if m else "", [(b, e, t.replace('""', '"')) for b, e, t in pat.findall(it)]))
    chosen = None
    for p in prefer:
        for name, ints in tiers:
            if name.lower() == p.lower():
                chosen = ints
    out = [(float(b), float(e), t.strip().strip("{}").strip()) for b, e, t in (chosen if chosen is not None else tiers[-1][1])]
    return [(b, e, t) for b, e, t in out if t.lower() not in SILENCE]


def bpe_token_events(words, tokenizer):

    ids, times = [], []
    for k, (b, e, w) in enumerate(words):
        piece = tokenizer(("" if k == 0 else " ") + w.lower(), add_special_tokens=False).input_ids
        if not piece:
            continue
        ids.extend(int(v) for v in piece)
        times.extend(b + (i + 0.5) * (e - b) / len(piece) for i in range(len(piece)))
    return ids, np.asarray(times, np.float64)


@torch.inference_mode()
def context_features(token_ids, model, device, batch=256):

    ids = torch.as_tensor(token_ids, dtype=torch.long); n = int(ids.numel()); first = min(CONTEXT_TOKENS, n)
    out = [model(input_ids=ids[:first][None].to(device), output_hidden_states=True, use_cache=False).hidden_states[FEATURE_LAYER][0].float().cpu()]
    for s in range(CONTEXT_TOKENS, n, batch):
        ctx = torch.stack([ids[i - CONTEXT_TOKENS:i + 1] for i in range(s, min(s + batch, n))])
        out.append(model(input_ids=ctx.to(device), output_hidden_states=True, use_cache=False).hidden_states[FEATURE_LAYER][:, -1].float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def build_tokens(textgrid, out_npz, model, tokenizer, device="cuda:0"):

    ids, times = bpe_token_events(textgrid_words(textgrid), tokenizer)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, ids=np.asarray(ids, np.int64), times=times, targets=context_features(ids, model, device))


def run_zscore(x):
    x = np.asarray(x, np.float32); mu, sd = x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)
    out = np.zeros_like(x); ok = sd > 1e-6
    out[:, ok] = (x[:, ok] - mu[ok]) / sd[ok]
    return np.nan_to_num(out)


def cifti_to_run(dtseries, out_npy):

    import nibabel as nib
    np.save(out_npy, run_zscore(np.asarray(nib.load(str(dtseries)).dataobj[:, :N_CORTEX], np.float32)))


def bandpass(x, tr, band):
    from scipy.signal import butter, filtfilt
    ny = 0.5 / tr; b, a = butter(2, [band[0] / ny, min(band[1] / ny, 0.99)], btype="band")
    return filtfilt(b, a, x, axis=0).astype(np.float32)


def causal_smooth(y, width):

    y = np.asarray(y, np.float32)
    if width <= 1:
        return y
    cum = np.vstack((np.zeros((1, y.shape[1])), np.cumsum(y, 0, dtype=np.float64)))
    pos = np.arange(len(y)); left = np.maximum(0, pos + 1 - width)
    return ((cum[pos + 1] - cum[left]) / (pos + 1 - left)[:, None]).astype(np.float32)


def fir_index(n_rows, times, delays, tr):

    pos = (np.asarray(times, np.float64)[:, None] + np.asarray(delays)[None]) / tr
    lo = np.floor(pos).astype(np.int64); keep = ((lo >= 0) & (lo + 1 < n_rows)).all(1)
    return lo[keep], lo[keep] + 1, (pos - np.floor(pos))[keep].astype(np.float32), keep


def subjects(dataset, lang):
    return [s.strip() for s in (DATA / dataset / f"subjects_{lang}.txt").read_text().split() if s.strip()]


def bold_file(dataset, subject, story):
    return DATA / dataset / "bold" / subject / f"{story}.npy"


def load_run(dataset, subject, story, vox=None):
    b = np.load(bold_file(dataset, subject, story), mmap_mode="r")
    b = np.asarray(b if vox is None else b[:, vox], np.float32)
    band = DATASETS[dataset]["band"]
    return bandpass(b, DATASETS[dataset]["tr"], band) if band else b


def load_tokens(dataset, lang, story, smooth):
    z = np.load(DATA / dataset / "tokens" / lang / f"{story}.npz")
    return z["times"].astype(np.float64), causal_smooth(z["targets"].astype(np.float32), smooth)


class Store:


    def __init__(self, dataset, lang, subs, stories, vox, device, delays=DELAYS, smooth=5):
        self.subjects, self.stories, self.vox, self.delays = list(subs), list(stories), np.asarray(vox), tuple(delays)
        self.device, tr = torch.device(device), DATASETS[dataset]["tr"]
        self.offset, rows, blocks = {}, 0, []
        for st in self.stories:
            self.offset[st] = rows; n = np.load(bold_file(dataset, self.subjects[0], st), mmap_mode="r").shape[0]; rows += n; blocks.append(n)
        self.data = torch.empty((len(self.subjects), rows, len(self.vox)), dtype=torch.float16, device=self.device)
        for i, s in enumerate(self.subjects):
            self.data[i] = torch.from_numpy(np.concatenate([load_run(dataset, s, st, self.vox) for st in self.stories])).to(self.device, torch.float16)
        self.tok = {}
        for st, n in zip(self.stories, blocks):
            times, y = load_tokens(dataset, lang, st, smooth)
            lo, hi, w, keep = fir_index(n, times, self.delays, tr)
            d = self.device
            self.tok[st] = {"lo": torch.from_numpy(lo + self.offset[st]).to(d), "hi": torch.from_numpy(hi + self.offset[st]).to(d),
                            "w": torch.from_numpy(w).to(d), "y": torch.from_numpy(y[keep]).to(d), "times": times[keep], "n": int(keep.sum())}

    def features(self, story, tok_idx, subj_idx):

        t = self.tok[story]; sj = subj_idx.unsqueeze(1)
        a, c = self.data[sj, t["lo"][tok_idx]].float(), self.data[sj, t["hi"][tok_idx]].float()
        return a + (c - a) * t["w"][tok_idx].unsqueeze(-1)


def feature_samples(store, stories, per_story):

    with torch.no_grad():
        sam = []
        for st in stories:
            n = store.tok[st]["n"]
            idx = torch.from_numpy(np.random.choice(n, min(n, per_story), replace=False)).to(store.device)
            sj = torch.from_numpy(np.random.randint(0, len(store.subjects), len(idx))).to(store.device)
            sam.append(store.features(st, idx, sj))
        return torch.cat(sam)


def feature_stats(samples):

    return samples.mean((0, 1)), samples.std((0, 1)).clamp_min(1e-3)


def training_rate(grids, tokenizer, continuous=False):

    words = bpe = 0; span = 0.0
    for g in grids:
        items = textgrid_words(g); ids, times = bpe_token_events(items, tokenizer); words += len(items)
        if continuous:
            toks = [w for _, _, w in items]
            sep = "" if sum(any("\u4e00" <= ch <= "\u9fff" for ch in w) for w in toks) > len(toks) / 2 else " "
            bpe += len(tokenizer(sep.join(toks))["input_ids"])
        else:
            bpe += len(ids)
        span += float(times[-1] - times[0])
    return words / span * (bpe / words)


def constant_clock(n_rows, tr, rate, delays=DELAYS):

    start, end = -min(delays), tr * (n_rows - 1) - max(delays)
    n = int(np.floor((end - start) * rate))
    return start + (np.arange(n) + 0.5) / rate, end
