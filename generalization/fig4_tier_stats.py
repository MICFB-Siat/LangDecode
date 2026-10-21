import json, itertools, numpy as np, pandas as pd
from langdecoder.paths import OUT
from common import DIRECTIONS

FLIPS = np.array(list(itertools.product([1, -1], repeat=6)))


def signflip_p(x):
    x = np.asarray(x, float); return float(np.mean((FLIPS * x).mean(1) >= x.mean() - 1e-12))


def bge(tag):
    return json.load(open(OUT / "texts" / f"score_{tag}.json"))["BGEScore_window_mean"]


def main():
    out = {}
    lesions = ("minus_Low", "minus_Mid", "minus_High", "minus_Random1", "minus_Random2", "minus_Random3")
    drop = {c: np.array([bge(f"lppc_{d}_whole") - bge(f"lppc_{d}_{c}") for d in DIRECTIONS]) for c in lesions}
    rnd = np.mean([drop[f"minus_Random{k}"] for k in (1, 2, 3)], 0)
    pd.DataFrame({"direction": DIRECTIONS, "Random": rnd, **{c.split("_")[1]: drop[c] for c in lesions[:3]}}).to_csv(OUT / "generalization" / "lppc_lesion_drop.csv", index=False)
    out["lppc_lesion"] = {"mean_drop": {c: float(v.mean()) for c, v in drop.items()}, "p_high_gt_low": signflip_p(drop["minus_High"] - drop["minus_Low"]),
                          "p_high_gt_mid": signflip_p(drop["minus_High"] - drop["minus_Mid"]), "p_high_gt_random": signflip_p(drop["minus_High"] - rnd)}
    conds = ("keep_Low", "keep_Mid", "keep_High", "keep_Random")
    ch = {c: np.mean([[bge(f"chen_{d}_{c}_{i}") for i in range(1, 7)] for d in ("en2zh", "zh2en")], 0) for c in conds}
    pd.DataFrame({c.split("_")[1]: v for c, v in ch.items()}).to_csv(OUT / "generalization" / "chen_tier_bge.csv", index=False)
    out["chen_tiers"] = {k: signflip_p(ch[a] - ch[b]) for k, (a, b) in {"High>Mid": ("keep_High", "keep_Mid"), "Mid>Low": ("keep_Mid", "keep_Low"),
                                                                     "High>Low": ("keep_High", "keep_Low"), "High>Random": ("keep_High", "keep_Random")}.items()}
    json.dump(out, open(OUT / "generalization" / "tier_decoding_stats.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
