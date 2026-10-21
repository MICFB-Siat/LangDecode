import argparse
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from langdecoder.paths import BLOOM
from .data import CONTEXT_TOKENS, FEATURE_LAYER, bpe_token_events, constant_clock, load_tokens, subjects, textgrid_words, training_rate
from .predict import feeds_for, tr_of, trajectory
from .train import load_decoder


class _Stop(Exception):
    pass


def layer_features(model, input_ids, **kw):

    box, block = {}, model.transformer.h[FEATURE_LAYER]

    def pre(module, args, kwargs):
        box["h"] = args[0] if args else kwargs["hidden_states"]; raise _Stop

    handle = block.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        model(input_ids=input_ids, output_hidden_states=True, use_cache=kw.pop("use_cache", False), **kw)
    except _Stop:
        pass
    finally:
        handle.remove()
    return box["h"]


@torch.inference_mode()
def decode(target, model, tokenizer, device, beam_width=12, candidates=96, cand_mean=None, cand_std=None, smooth=5,
           allowed=None, score_fn=None, verbose=True):

    target_t = F.normalize(torch.from_numpy(np.asarray(target, np.float32)).to(device), dim=1)
    mu = torch.from_numpy(np.asarray(cand_mean, np.float32)).to(device) if cand_mean is not None else None
    sd = torch.from_numpy(np.asarray(cand_std, np.float32)).to(device) if cand_std is not None else None
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    blocked = {v for v in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) if v is not None}
    hw = max(smooth - 1, 0)
    beam = [([], 0.0, None)]
    for pos in range(len(target_t)):
        if not beam[0][0]:
            contexts = torch.full((len(beam), 1), int(bos), dtype=torch.long, device=device)
        else:
            cl = min(CONTEXT_TOKENS, len(beam[0][0]))
            contexts = torch.as_tensor([s[0][-cl:] for s in beam], dtype=torch.long, device=device)
        logits = model(input_ids=contexts, use_cache=False).logits[:, -1].float()
        if allowed is not None:
            logits = logits.masked_fill(~allowed[:logits.shape[-1]].view(1, -1), float("-inf"))
        top_ids = torch.topk(F.log_softmax(logits, dim=-1), k=candidates, dim=-1)[1].tolist()
        ext = [(si, int(t)) for si in range(len(beam)) for t in top_ids[si] if t not in blocked]
        state_idx = torch.as_tensor([si for si, _ in ext], device=device)
        feat_ids = torch.as_tensor([beam[si][0][-CONTEXT_TOKENS:] + [t] for si, t in ext], dtype=torch.long, device=device)
        if not beam[0][0]:
            hidden = layer_features(model, feat_ids)[:, -1].float()
        else:
            pre = torch.as_tensor([s[0][-CONTEXT_TOKENS:] for s in beam], dtype=torch.long, device=device)
            past = tuple((k.index_select(0, state_idx), v.index_select(0, state_idx)) for k, v in model(input_ids=pre, use_cache=True).past_key_values)
            attn = torch.ones((len(ext), pre.shape[1] + 1), dtype=torch.long, device=device)
            hidden = layer_features(model, feat_ids[:, -1:], past_key_values=past, attention_mask=attn, use_cache=True)[:, -1].float()
        hist = [0 if (s[2] is None or hw == 0) else min(hw, s[2].shape[0]) for s in beam]
        cand = hidden
        if any(hist):
            earlier = torch.stack([beam[si][2][-hw:].sum(0) if hist[si] else torch.zeros(hidden.shape[1], device=device) for si in range(len(beam))])
            denom = torch.as_tensor([float(hist[si] + 1) for si, _ in ext], device=device)
            has = torch.as_tensor([hist[si] > 0 for si, _ in ext], device=device)
            cand = torch.where(has[:, None], (hidden + earlier[state_idx]) / denom[:, None], hidden)
        if mu is not None:
            cand = (cand - mu) / sd
        sim = (F.normalize(cand, dim=1) @ target_t[pos]).tolist()
        if score_fn is not None:
            sim = score_fn(sim)
        scores = [beam[si][1] + float(s) for (si, _), s in zip(ext, sim)]
        new = []
        for i in sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:beam_width]:
            si, t = ext[i]; old = beam[si][2]
            prev = old[-hw:] if (old is not None and hw) else hidden[i:i + 1][:0]
            new.append((beam[si][0] + [t], scores[i], torch.cat([prev, hidden[i:i + 1]], 0)[-smooth:]))
        beam = new
        if verbose and ((pos + 1) % 256 == 0 or pos + 1 == len(target_t)):
            print(json.dumps({"decoded_tokens": pos + 1, "total_tokens": len(target_t)}), flush=True)
    return beam[0][0]


def lexicon_mask(grids, tokenizer, vocab_size, device):

    ids = set()
    for g in grids:
        ids.update(bpe_token_events(textgrid_words(g), tokenizer)[0])
    ok = np.zeros(vocab_size, bool); ok[sorted(ids)] = True
    ok[[s for s in tokenizer.all_special_ids if s < vocab_size]] = False
    return torch.from_numpy(ok).to(device)


def prepare(a, device):

    net, ck = load_decoder(a.decoder, device)
    tokenizer = AutoTokenizer.from_pretrained(str(BLOOM))
    grids = sorted(glob.glob(a.train_grids))
    target_subjects = a.target_subjects.split(",") if a.target_subjects else subjects(a.dataset, a.lang)
    feeds = feeds_for(ck, a.dataset, a.story, a.feed, target_subjects)
    n_rows, tr = min(len(v) for v in feeds.values()), tr_of(a.dataset)
    rate = training_rate(grids, tokenizer, continuous=a.clock_continuous)
    if a.clock == "rate":
        times, end = constant_clock(n_rows, tr, rate, ck["delays"])
    else:
        times, end = load_tokens(a.dataset, a.lang, a.story, 1)[0], tr * (n_rows - 1) - max(ck["delays"])
    pred, kept = trajectory(net, ck, feeds, times, tr, device=device)
    return pred, times[kept], end, rate, ck, tokenizer, grids


def main(argv=None, score_fn=None, tag="brain"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True); ap.add_argument("--dataset", default="lppc"); ap.add_argument("--lang", required=True)
    ap.add_argument("--story", default="story09"); ap.add_argument("--feed", choices=["own", "group"], default="own")
    ap.add_argument("--target-subjects", default=""); ap.add_argument("--train-grids", required=True)
    ap.add_argument("--clock", choices=["rate", "tokens"], default="rate"); ap.add_argument("--clock-continuous", action="store_true",
                                                                                         help="rate from running-text tokenization (Chinese)")
    ap.add_argument("--mask", choices=["none", "lexicon"], default="none")
    ap.add_argument("--beam-width", type=int, default=12); ap.add_argument("--candidates", type=int, default=96)
    ap.add_argument("--smooth", type=int, default=5); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv); device = torch.device(a.device)
    pred, times, end, rate, ck, tokenizer, grids = prepare(a, device)
    model = AutoModelForCausalLM.from_pretrained(str(BLOOM), torch_dtype=torch.bfloat16).to(device).eval()
    allowed = lexicon_mask(grids, tokenizer, model.config.vocab_size, device) if a.mask == "lexicon" else None
    ids = decode(pred, model, tokenizer, device, a.beam_width, a.candidates, ck["ymean"].numpy(), ck["ystd"].numpy(), a.smooth,
                 allowed, score_fn=score_fn(a.seed) if score_fn else None)
    out = {"condition": tag, "story": a.story, "beam_width": a.beam_width, "candidates": a.candidates, "mask": a.mask, "clock": a.clock,
           "timing": {"bpe_per_second": rate, "token_times": [float(t) for t in times], "end_seconds": float(end), "n_generated_bpe": len(ids)},
           "generated_pieces": [tokenizer.decode([i], skip_special_tokens=True) for i in ids],
           "prediction": tokenizer.decode(ids, skip_special_tokens=True)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out


if __name__ == "__main__":
    main()
