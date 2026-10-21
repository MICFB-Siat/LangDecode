import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from langdecoder.paths import BGE_M3, BLOOM
from .data import textgrid_words


def windows(start, end, width=20):
    half = width // 2
    return [(c - half, c + half) for c in range(start + half, end - half + 1)]


def segment(words, times, cuts):
    return [" ".join(words[(times >= s) & (times < e)].tolist()) for s, e in cuts]


def bge_vectors(texts, device="cuda:0", batch=16):
    from FlagEmbedding import BGEM3FlagModel
    m = BGEM3FlagModel(str(BGE_M3), use_fp16=device.startswith("cuda"), device=device)
    v = np.asarray(m.encode(texts, batch_size=batch, max_length=512, return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"], np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def temporal_nulls(cand, ref, n_null=2000, seed=20260820):
    rng = np.random.default_rng(seed); n = len(cand); paired = float(np.mean(np.sum(cand * ref, 1)))
    shifts = np.arange(20, n - 19); perm, circ = [], []
    for _ in range(n_null):
        perm.append(float(np.mean(np.sum(cand * ref[rng.permutation(n)], 1))))
        circ.append(float(np.mean(np.sum(cand * np.roll(ref, int(rng.choice(shifts)), 0), 1))))

    def rep(v):
        v = np.asarray(v); return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "one_sided_p": float((1 + np.sum(v >= paired)) / (len(v) + 1))}
    return paired, {"permuted_windows": rep(perm), "circular_shift_at_least_20_s": rep(circ)}


def lexical_tokens(text, language):
    if language == "zh":
        return [c for c in text if "\u3400" <= c <= "\u9fff"]
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def bleu1(ref, pred):
    if not ref or not pred:
        return 0.0
    rc = Counter(ref); clipped = sum(min(c, rc[t]) for t, c in Counter(pred).items())
    return clipped / len(pred) * min(1.0, float(np.exp(1.0 - len(ref) / len(pred))))


def lexical_summary(cands, refs, language):
    from jiwer import wer
    from nltk.translate.meteor_score import meteor_score
    pairs = [(c, r) for c, r in ((lexical_tokens(c, language), lexical_tokens(r, language)) for c, r in zip(cands, refs)) if r]
    return {"WER": float(np.mean([wer(" ".join(r), " ".join(c)) for c, r in pairs])),
            "BLEU_1": float(np.mean([bleu1(r, c) for c, r in pairs])),
            "METEOR": float(np.mean([meteor_score([r], c) for c, r in pairs]))}


def score(generation, transcript, language="en", device="cuda:0", start=10, width=20, n_null=2000):
    g = json.loads(Path(generation).read_text(encoding="utf-8")); tm = g["timing"]
    tok = AutoTokenizer.from_pretrained(str(BLOOM))
    ids = tok(g["prediction"], add_special_tokens=False).input_ids
    rec = np.asarray(tm["token_times"], np.float64)
    if len(rec) == len(ids):
        times = rec
    else:
        edges = np.cumsum([0] + [len(x) for x in g["generated_pieces"]]); text = g["prediction"]
        ct = np.full(max(edges[-1], len(text)) + 1, rec[-1])
        for k in range(len(rec)):
            ct[edges[k]:edges[k + 1]] = rec[k]
        times = np.asarray([ct[min(a, len(ct) - 1)] for a, _ in tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]])
    items = textgrid_words(transcript)
    rw, rt = np.asarray([w.lower() for _, _, w in items]), np.asarray([(b + e) / 2 for b, e, _ in items])
    cuts = windows(int(start), int(tm["end_seconds"]), width)
    refs = segment(rw, rt, cuts)
    cands = [tok.decode(np.asarray(ids)[(times >= s) & (times < e)].tolist(), skip_special_tokens=True) for s, e in cuts]
    ok = [bool(r.strip()) and bool(c.strip()) for r, c in zip(refs, cands)]
    refs, cands = [r for r, k in zip(refs, ok) if k], [c for c, k in zip(cands, ok) if k]
    rv, cv = bge_vectors(refs, device), bge_vectors(cands, device)
    paired, nulls = temporal_nulls(cv, rv, n_null)
    return {"BGEScore": paired, "per_window": np.sum(cv * rv, 1).tolist(), **lexical_summary(cands, refs, language), **nulls, "n_windows": len(refs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", required=True); ap.add_argument("--transcript", required=True)
    ap.add_argument("--language", default="en", choices=["en", "fr", "zh"]); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window", type=int, default=20); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = score(a.generation, a.transcript, a.language, a.device, width=a.window)
    Path(a.out).write_text(json.dumps(r, indent=1)); print(json.dumps({k: r[k] for k in ("BGEScore", "WER", "BLEU_1", "METEOR", "n_windows")}))


if __name__ == "__main__":
    main()
