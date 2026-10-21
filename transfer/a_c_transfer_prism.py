import glob, json, re
import pandas as pd
from common import F5

rows = []
for f in glob.glob(str(F5 / "text" / "score_*_p*_*_sub-*.json")):
    m = re.search(r"score_(lppc|chen)_(\w+?)_p(\d+)_(base|xl)_(sub-\w+)\.json$", f)
    if m:
        rows.append(dict(dataset=m[1], lang=m[2], pct=int(m[3]), model={"base": "Baseline", "xl": "langdecoder"}[m[4]], sub=m[5],
                         bge=json.load(open(f))["BGEScore_window_mean"]))
D = pd.DataFrame(rows)
for (ds, lang), g in D.groupby(["dataset", "lang"]):
    W = g.pivot_table(index="sub", columns=["pct", "model"], values="bge")
    W.columns = [f"{p}% {m}" for p, m in W.columns]
    W.reset_index().to_csv(F5 / f"prism_fig5_{ds}_{lang}.csv", index=False, float_format="%.5f")
D.groupby(["dataset", "lang", "pct", "model"]).bge.agg(["count", "mean", "median"]).round(4).to_csv(F5 / "fig5ac_summary.csv")
print(D.groupby(["dataset", "lang", "pct", "model"]).bge.mean().unstack().round(4))
