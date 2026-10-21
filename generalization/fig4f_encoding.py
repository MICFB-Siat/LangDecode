import sys, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from langdecoder.paths import OUT, LANGS
from langdecoder.surface import parcel_labels, parcel_mean
from langdecoder.stats import holm
from common import shared_map_vertex, thirds
from encoding import lss_weights, zscore, encode_participant


def sentence_betas(lang, subject):
    ev = pd.read_csv(OUT / "events" / f"lppc_{lang}.csv"); out = {}
    for s, e in ev.groupby("story"):
        bold = np.load(OUT / "bold" / subject / f"bold_story{int(s):02d}.npy").astype(np.float32)
        out[int(s)] = lss_weights(e.glm_onset.to_numpy(), e.glm_offset.to_numpy(), len(bold)) @ zscore(bold)
    return out


def features(lang):
    ev = pd.read_csv(OUT / "events" / f"lppc_{lang}.csv"); f = np.load(OUT / "features" / f"lppc_{lang}_bloom20_sentence.npy")
    return {int(s): f[(ev.story == s).to_numpy()] for s in sorted(ev.story.unique())}


def fit(lang, subject):
    subs = {L: open(OUT / f"subjects_{L.lower()}.txt").read().split() for L in LANGS}; F = {L: features(L) for L in LANGS}
    own = sentence_betas(lang, subject); groups = {}
    for L in LANGS:
        members = [s for s in subs[L] if s != subject]; acc = [sentence_betas(L, s) for s in members]
        groups[L] = {st: np.mean([a[st] for a in acc], 0) for st in F[L]}
    z = encode_participant(F[lang], own, F, groups, parcel_labels("destrieux"))
    (OUT / "encoding" / lang).mkdir(parents=True, exist_ok=True); np.savez(OUT / "encoding" / lang / f"{subject}.npz", **z)


def summary():
    par = parcel_labels(); geo = parcel_mean(np.mean([shared_map_vertex(p) for p in ("CN_EN", "EN_FR")], 0)); t = thirds(geo)
    tier = {n: np.isin(par, np.flatnonzero(t == k) + 1) for k, n in enumerate(("Low", "Mid", "High"))}; rows = []
    for L in LANGS:
        for s in open(OUT / f"subjects_{L.lower()}.txt").read().split():
            z = np.load(OUT / "encoding" / L / f"{s}.npz"); cross = np.mean([z[T] for T in LANGS if T != L], 0)
            for n, m in tier.items():
                c, w = np.tanh(np.nanmean(cross[m])), np.tanh(np.nanmean(z[L][m])); rows.append(dict(participant=s, source=L, level=n, cross_r=c, within_r=w, transfer_ratio=c / w if w > 0 else np.nan))
    D = pd.DataFrame(rows); D.to_csv(OUT / "generalization" / "fig4f_encoding_persub.csv", index=False)
    W = D.pivot_table(index="participant", columns="level", values="cross_r")[["Low", "Mid", "High"]]; W.to_csv(OUT / "generalization" / "fig4f_prism.csv", index=False)
    pairs = (("Mid", "Low"), ("High", "Mid"), ("High", "Low")); st = {"n": len(W)}
    for side in ("greater", "two-sided"):
        p = [wilcoxon(W[a], W[b], alternative=side).pvalue for a, b in pairs]; st[side] = dict(zip([f"{a}>{b}" for a, b in pairs], map(float, holm(p))))
    json.dump(st, open(OUT / "generalization" / "fig4f_stats.json", "w"), indent=1); print(json.dumps(st, indent=1))


if __name__ == "__main__":
    fit(sys.argv[2], sys.argv[3]) if sys.argv[1] == "fit" else summary()
