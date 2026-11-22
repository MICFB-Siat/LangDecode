import pandas as pd
from scipy.stats import wilcoxon
from langdecoder.paths import OUT

FS = OUT / "nebula" / "fewshot_tier"; D = pd.read_csv(FS / "tier_per_listener.csv")
lab = {"T1": "Low", "T2": "Mid", "T3": "High", "R1": "Random"}; S, G = [], {}
for (fr, k), g in D.groupby(["frac", "tier"]):
    b = g[g.init == "smn"].set_index("sub").acc
    for init in ("cn", "fr"):
        x = g[g.init == init].set_index("sub").acc; kk = b.index.intersection(x.index); d = x[kk] - b[kk]; G[(fr, k, init)] = d
        S.append(dict(frac=fr, group=lab[k], init=init, n=len(kk), baseline=b[kk].mean(), pretrained=x[kk].mean(), gain=d.mean(), p=wilcoxon(d).pvalue))
    for init in ("cn", "fr"):
        if k == "T3" and (fr, "T1", init) in G:
            d = (G[(fr, "T3", init)] - G[(fr, "T1", init)]).dropna(); S.append(dict(frac=fr, group="High-Low", init=init, n=len(d), gain=d.mean(), p=wilcoxon(d).pvalue))
pd.DataFrame(S).round(5).to_csv(FS / "ed_transfer_gain_summary.csv", index=False)
pd.DataFrame({f"{int(fr * 100)}% {'Chinese' if i == 'cn' else 'French'}-pretrained gain {lab[k]}": d for (fr, k, i), d in G.items()}).sort_index(axis=1).to_csv(
    FS / "prism_ed_transfer_gain.csv", float_format="%.5f")
print(pd.DataFrame(S).round(4).to_string(index=False))
