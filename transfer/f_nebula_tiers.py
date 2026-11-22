import numpy as np, pandas as pd
from common import NEB

FS = NEB / "fewshot_tier"; zz = np.load(NEB / "zeroshot_cn_pred.npz"); TK, TG = list(zz["targets_keys"]), zz["targets"]
norm = lambda A: (A - A.mean(0)) / (np.linalg.norm(A - A.mean(0), axis=1, keepdims=True) + 1e-8)
rows = []
for fr in ("0.3", "0.5", "0.7"):
    for k in ("T1", "T2", "T3", "R1"):
        for init in ("smn", "cn", "fr"):
            f0, f1 = (FS / f"French_{k}_{init}_f{fr}_fold{f}_pred.npz" for f in (0, 1))
            if not (f0.exists() and f1.exists()):
                continue
            A, B = np.load(f0), np.load(f1)
            for s in sorted({x.split("|")[0] for x in A.files} & {x.split("|")[0] for x in B.files}):
                ps = np.concatenate([A[f"{s}|p"], B[f"{s}|p"]]); X = np.concatenate([A[f"{s}|x"] - A[f"{s}|x"].mean(0), B[f"{s}|x"] - B[f"{s}|x"].mean(0)])
                o = np.argsort(ps); ps, X = ps[o], X[o]; C = norm(X) @ norm(np.stack([TG[TK.index(f"French|{int(p)}")] for p in ps])).T
                rows.append(dict(frac=float(fr), tier=k, init=init, sub=s, acc=float(np.mean([(C[i] < C[i, i]).sum() / (len(C) - 1) for i in range(len(C))]))))
D = pd.DataFrame(rows); D.to_csv(FS / "tier_per_listener.csv", index=False)
W = D.pivot_table(index="sub", columns=["frac", "tier", "init"], values="acc"); lab = {"T1": "Low", "T2": "Mid", "T3": "High", "R1": "Random"}
pd.DataFrame({lab[t]: W[(0.3, t, "cn")] for t in ("T1", "T2", "T3")}).to_csv(FS / "prism_fig5f_langdecoder_ChinesePretrained_30pct.csv", index=False, float_format="%.5f")
pd.DataFrame({f"{int(fr * 100)}% {m} {lab[t]}": (W[(fr, t, "smn")] if m == "Baseline" else (W[(fr, t, "cn")] + W[(fr, t, "fr")]) / 2)
              for fr in (0.3, 0.5, 0.7) for m in ("Baseline", "langdecoder") for t in lab}).to_csv(FS / "prism_nebula_tier_accuracy_all.csv", index=False, float_format="%.5f")
print(W.mean().unstack("init").round(4))
