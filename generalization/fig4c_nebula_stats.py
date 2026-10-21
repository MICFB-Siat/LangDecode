import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from langdecoder.paths import OUT


def main():
    D = pd.read_csv(OUT / "generalization" / "nebula_zeroshot_wc_cn.csv")
    U = pd.concat([pd.read_csv(OUT / "generalization" / f"nebula_zeroshot_untrained_s{k}.csv") for k in (0, 1, 2)]).groupby(["sub", "cond"]).identification.mean()
    cols, out = {}, {}
    for cond, lab in (("intact", "CN->FR"), ("foreign", "CN->EN")):
        d = D[(D.keep == "all") & (D.cond == cond)].set_index("sub").identification; u = U.xs(cond, level="cond").reindex(d.index)
        cols[f"{lab} Untrained"], cols[f"{lab} langdecoder"] = u.to_numpy(), d.to_numpy()
        out[lab] = {"n": len(d), "p_two_sided": float(wilcoxon(d - u).pvalue), "median_untrained": float(u.median()), "median_langdecoder": float(d.median())}
        p = D[D.cond == cond].pivot_table(index="sub", columns="keep", values="identification").dropna(); rnd = p[["Random1", "Random2", "Random3"]].mean(1)
        p.assign(Random=rnd)[["Low", "Mid", "High", "Random"]].to_csv(OUT / "generalization" / f"nebula_tiers_{cond}.csv")
        out[lab]["tiers"] = {"n": len(p), "p_mid_gt_low": float(wilcoxon(p.Mid, p.Low, alternative="greater").pvalue),
                             "p_high_gt_mid": float(wilcoxon(p.High, p.Mid, alternative="greater").pvalue), "p_high_gt_random": float(wilcoxon(p.High, rnd, alternative="greater").pvalue)}
    pd.DataFrame({k: pd.Series(v) for k, v in cols.items()}).to_csv(OUT / "generalization" / "fig4c_prism.csv", index=False)
    json.dump(out, open(OUT / "generalization" / "nebula_stats.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
