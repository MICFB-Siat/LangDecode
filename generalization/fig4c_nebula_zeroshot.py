import argparse, glob, re, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langdecoder.paths import OUT, NEBULA, BLOOM, N_CORTEX
from langdecoder.surface import parcel_labels, parcel_mean
from common import PAIRS
import crosslang as X


@torch.no_grad()
def bloom_targets(transcripts, norm):
    tok = AutoTokenizer.from_pretrained(BLOOM); lm = AutoModelForCausalLM.from_pretrained(BLOOM).to(X.DEV).eval(); out = {}
    for _, r in transcripts.iterrows():
        words = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", str(r.text).lower()); ids = []
        for k, w in enumerate(words): ids += tok(("" if k == 0 else " ") + w, add_special_tokens=False).input_ids
        ids = torch.tensor(ids); h = [lm(ids[:15][None].to(X.DEV), output_hidden_states=True).hidden_states[20][0]]
        h += [lm(ids[i - 15:i + 1][None].to(X.DEV), output_hidden_states=True).hidden_states[20][0, -1:] for i in range(15, len(ids))]
        y = X.causal_smooth(torch.cat(h).float().cpu().numpy(), 5); out[(r.language, int(r.passage))] = ((y - norm["ymean"]) / norm["ystd"]).mean(0)
    return out


def tier_masks():
    par = parcel_labels(); gp = parcel_mean(np.mean([np.load(OUT / "semantic_geometry" / f"semantic_map_9story_{p}.npy") for p in PAIRS], 0))
    ok = np.flatnonzero(np.isfinite(gp)); parts = np.array_split(ok[np.argsort(gp[ok], kind="stable")], 3); rng = np.random.default_rng(20261005); m = {"all": None}
    for k, ps in enumerate(parts): m[("Low", "Mid", "High")[k]] = np.isin(par, ps + 1)
    for k, ps in enumerate(parts): m[f"Random{k + 1}"] = np.isin(par, rng.choice(ok, len(ps), replace=False) + 1)
    return m


def identification(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0); P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8; Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
    C = P @ Y.T; return float(np.mean([(C[i] < C[i, i]).sum() / (len(C) - 1) for i in range(len(C))]))


@torch.no_grad()
def passage_prediction(net, norm, run, onset, dur, keep):
    t = onset + np.arange(0, dur, 0.5); pos = (t[:, None] + X.DELAYS[None]) / X.TR; pos = pos[pos.max(1) < len(run) - 1]
    lo = np.floor(pos).astype(np.int64); return X.predict(net, norm, run, lo, (pos - lo).astype(np.float32), keep).mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--untrained", type=int); a = ap.parse_args()
    net, norm = X.load_decoder("wc_cn", untrained_seed=a.untrained)
    TG = bloom_targets(pd.read_csv(NEBULA / "whisper_transcripts.tsv", sep="\t"), norm); masks = tier_masks() if a.untrained is None else {"all": None}; rows = []
    for f in sorted(glob.glob(str(NEBULA / "passage_series" / "*.npz"))):
        z = np.load(f); ev = z["events"]; runs = {r: X.zscore(z[f"run{r}"]) for r in (1, 2, 3)}; native = str(ev[ev["cond"] == "intact"]["lang"][0])
        if native != "French": continue
        for keep_name, keep in masks.items():
            for cond in ("intact", "foreign"):
                e = np.sort(ev[ev["cond"] == cond], order="passage"); lang = str(e["lang"][0])
                P = np.stack([passage_prediction(net, norm, runs[int(x["run"])], float(x["onset"]), float(x["duration"]), keep) for x in e])
                rows.append({"sub": f.split("/")[-1][:-4], "keep": keep_name, "cond": cond, "heard_lang": lang,
                             "identification": identification(P, np.stack([TG[(lang, int(p))] for p in e["passage"]]))})
    tag = "wc_cn" if a.untrained is None else f"untrained_s{a.untrained}"; pd.DataFrame(rows).to_csv(OUT / "generalization" / f"nebula_zeroshot_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
