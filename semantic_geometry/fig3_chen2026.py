import h5py
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from langdecoder.paths import CHEN
from _common import OUT3, EVENTS, lss_weights, zscore_time, corr_rdm, fz

O = OUT3 / "chen2026"; SUBS = [f"sub-UCBS0{i}" for i in range(1, 7)]; LANGS2 = ("en", "zh"); IU = np.triu_indices(24, 1)


def anchors(ev):
    prog = ev.glm_onset.to_numpy() / float(ev.glm_offset.max()); out, last = [], -1
    for k in range(24):
        cand = np.arange(last + 1, len(ev) - (23 - k)); j = int(cand[np.argmin(np.abs(prog[cand] - (k + 0.5) / 24))]); out.append(j); last = j
    return np.array(out)


def rdms(sub, lang):
    ev_all = pd.read_csv(EVENTS / f"chen_{lang}_pause0p65.csv"); out = []
    for st in sorted(ev_all.story.unique()):
        ev = ev_all[ev_all.story == st].reset_index(drop=True)
        with h5py.File(CHEN / "preproc_data" / sub / f"{st}.hf5", "r") as h: x = np.asarray(h[lang], np.float32)
        valid = np.isfinite(x).all(0) & (x.std(0) > 1e-6); mf = CHEN / "valid_masks" / sub / f"{st}.npz"
        if mf.exists(): valid &= np.load(mf)[lang].astype(bool)
        b = lss_weights(ev, len(x))[anchors(ev)] @ zscore_time(x, valid); out.append(corr_rdm(b[:, valid]))
    return np.stack(out)


def main():
    O.mkdir(parents=True, exist_ok=True); D = {(l, s): rdms(s, l) for l in LANGS2 for s in SUBS}; ns = len(D[("en", SUBS[0])]); rows = []
    for a in SUBS:
        for b in SUBS:
            r = [spearmanr(D[("en", a)][k][IU], D[("zh", b)][k][IU])[0] for k in range(ns)]
            rows.append({"en_participant": a, "zh_participant": b, "within_person": a == b, "similarity": float(np.tanh(np.nanmean(fz(np.array(r)))))})
    pd.DataFrame(rows).to_csv(O / "chen2026_rdm_similarity.csv", index=False, float_format="%.5f")
    G = {l: np.mean([D[(l, s)] for s in SUBS], 0) for l in LANGS2}
    x = np.concatenate([G["en"][k][IU] for k in range(ns)]); y = np.concatenate([G["zh"][k][IU] for k in range(ns)]); rng = np.random.default_rng(1); rho = spearmanr(x, y)[0]
    null = [spearmanr(x, np.concatenate([G["zh"][k][np.ix_(p, p)][IU] for k in range(ns) for p in [rng.permutation(24)]]))[0] for _ in range(1000)]
    pd.DataFrame({"EN_neural_distance": x, "ZH_neural_distance": y}).to_csv(O / "fig3e_pairs_Chen2026.csv", index=False, float_format="%.5f")
    print({"event_pair_rho": rho, "p_perm": (1 + np.sum(np.array(null) >= rho)) / 1001})


if __name__ == "__main__":
    main()
