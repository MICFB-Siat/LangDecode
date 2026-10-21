import re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from langdecoder.paths import DATA
from langdecoder.stats import holm
from _common import OUT3, LANGS, cohort

O = OUT3 / "semantic_attributes"; B = OUT3 / "betas"; NORMS = DATA / "norms"; SHOW = ("concreteness", "socialness", "arousal"); N_NULL = 200
STOP = set(("a an the and or but if of to in on at by for with from as is are was were be been being am do does did have has had it its this that "
            "these those there here not no so than then too very just can could would should will shall may might must i me my we us our you your he "
            "him his she her they them their what which who whom whose when where why how all any some each every such only own same other into over "
            "under again further once about against between through during before after above below up down out off also s t don").split())
SUFFIX = (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("ly", ""), ("'s", ""), ("n't", ""))


def norms():
    c = pd.read_csv(NORMS / "Concreteness_ratings_Brysbaert_et_al_BRM.txt", sep="\t"); c = c[c.Bigram == 0]
    s = pd.read_csv(NORMS / "SocialnessNorms_DiveicaPexmanBinney2021.csv"); w = pd.read_csv(NORMS / "Ratings_Warriner_et_al.csv")
    return {"concreteness": dict(zip(c.Word.str.lower(), c["Conc.M"])), "socialness": dict(zip(s.Word.str.lower(), s.Mean)),
            "arousal": dict(zip(w.Word.str.lower(), w["A.Mean.Sum"]))}


def lookup(w, d):
    if w in d: return d[w]
    for suf, rep in SUFFIX:
        if w.endswith(suf) and len(w) - len(suf) >= 3 and w[:-len(suf)] + rep in d: return d[w[:-len(suf)] + rep]


def scores(texts, d):
    out = []
    for t in texts:
        ws = [w.strip("'") for w in re.findall(r"[a-z']+", t.lower().replace("’", "'")) if w.strip("'") and w not in STOP]
        v = [x for x in (lookup(w, d) for w in ws) if x is not None]; out.append(np.mean(v) if v else np.nan)
    return np.array(out, float)


def resid(sc, story, nw, dur):
    r = np.full(len(sc), np.nan)
    for s in np.unique(story):
        i = (story == s) & np.isfinite(sc); z = np.column_stack([np.ones(i.sum()), nw[i], np.log(dur[i] + 0.1)])
        r[i] = sc[i] - z @ np.linalg.lstsq(z, sc[i], rcond=None)[0]
    return (r - np.nanmean(r)) / np.nanstd(r)


def thirds(sc, story):
    lab = np.zeros(len(sc), int)
    for s in np.unique(story):
        i = np.flatnonzero((story == s) & np.isfinite(sc)); q1, q2 = np.percentile(sc[i], [100 / 3, 200 / 3]); lab[i[sc[i] <= q1]] = -1; lab[i[sc[i] >= q2]] = 1
    return lab


def clean(x, story, nw, dur):

    x = x.astype(np.float32).copy()
    for s in np.unique(story):
        i = story == s; x[i] = (x[i] - x[i].mean(0)) / np.maximum(x[i].std(0), 1e-6)
    z = np.column_stack([(story == s).astype(np.float32) for s in np.unique(story)] +
                        [(v - v.mean()) / v.std() for v in (nw.astype(float), np.log(dur + 0.1)) if v.std() > 0]).astype(np.float32)
    return x - z @ np.linalg.lstsq(z, x, rcond=None)[0]


def weights(lab):
    w = np.zeros(len(lab), np.float32); w[lab == 1] = 1 / (lab == 1).sum(); w[lab == -1] = -1 / (lab == -1).sum(); return w


def unit(m):
    return m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-12)


def main():
    O.mkdir(parents=True, exist_ok=True); g = {L: np.load(B / f"group_{L}.npz", allow_pickle=True) for L in LANGS}
    common = set.intersection(*(set(g[L]["sentence_index"]) for L in LANGS)); rows = {L: np.flatnonzero(np.isin(g[L]["sentence_index"], list(common))) for L in LANGS}
    story = g["EN"]["story"][rows["EN"]]; dur = g["EN"]["dur"][rows["EN"]].astype(float)
    nw = {L: (np.array([len(t.replace(" ", "")) for t in g[L]["text"]]) if L == "CN" else g[L]["n_words"])[rows[L]] for L in LANGS}
    lab = {a: thirds(resid(scores(g["EN"]["text"][rows["EN"]], d), story, nw["EN"], dur), story) for a, d in norms().items()}
    Wa = np.stack([weights(lab[a]) for a in SHOW], 1); rng = np.random.default_rng(0); Wn = np.zeros((len(story), N_NULL), np.float32)
    for t in range(N_NULL):
        sc = np.zeros(len(story))
        for s in np.unique(story): i = np.flatnonzero(story == s); sc[i] = rng.permutation(len(i))
        Wn[:, t] = weights(thirds(sc, story))
    gu = {L: unit(clean(g[L]["beta"][rows[L]], story, nw[L], g[L]["dur"][rows[L]].astype(float)).T @ Wa) for L in LANGS}
    rec = []
    for L in LANGS:
        others = [M for M in LANGS if M != L]
        for sub in cohort()[L]:
            x = clean(np.load(B / f"{L}_{sub}.npy")[rows[L]], story, nw[L], g[L]["dur"][rows[L]].astype(float)); ar, nu = unit(x.T @ Wa), unit(x.T @ Wn)
            for k, a in enumerate(SHOW):
                null = np.mean([nu.T @ gu[M][:, k] for M in others], 0)
                rec.append({"language": L, "participant": sub, "attribute": a, "ps": float(np.mean([ar[:, k] @ gu[M][:, k] for M in others])),
                            "random_control": float(np.median(null))})
    d = pd.DataFrame(rec); d.to_csv(O / "ps_participants.csv", index=False, float_format="%.5f")
    wide = {}
    for a in SHOW: wide[a.capitalize()] = d[d.attribute == a].ps.to_numpy(); wide[f"{a.capitalize()} random"] = d[d.attribute == a].random_control.to_numpy()
    pd.DataFrame(wide).to_csv(O / "fig3d_prism.csv", index=False, float_format="%.5f")
    p = [wilcoxon(d[d.attribute == a].ps - d[d.attribute == a].random_control, alternative="greater").pvalue for a in SHOW]
    pd.DataFrame({"attribute": SHOW, "n": [int((d.attribute == a).sum()) for a in SHOW], "p": p, "p_holm": holm(p)}).to_csv(O / "fig3d_stats.csv", index=False)


if __name__ == "__main__":
    main()
