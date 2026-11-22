import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from _common import OUT3, LANGS, PAIRS, N_STORY, cohort, events, run_files, story_betas, lppc_anchors, corr_rdm, cmds, fz

O = OUT3 / "anchors"; IU = np.triu_indices(24, 1)


def subject_rdms(lang, sub, anchors):
    ev_all, files, r = events(lang), run_files(sub), np.zeros((N_STORY, 24, 24), np.float32)
    for s in range(1, N_STORY + 1):
        ev = ev_all[ev_all.story == s].reset_index(drop=True); b, valid = story_betas(files[s - 1], ev, rows=anchors[s])
        r[s - 1] = corr_rdm(b[:, valid])
    return r


def main():
    O.mkdir(parents=True, exist_ok=True); c, anc = cohort(), lppc_anchors()
    S = {L: np.stack(Parallel(n_jobs=8)(delayed(subject_rdms)(L, s, anc) for s in c[L])) for L in LANGS}
    G = {L: S[L].mean(0) for L in LANGS}; np.savez(O / "group_rdms.npz", **G, **{f"subj_{L}": S[L] for L in LANGS})
    X = {L: cmds(G[L][0].astype(float)) for L in LANGS}; ref = X["CN"] - X["CN"].mean(0); al = {}
    for L in LANGS:
        Y = X[L] - X[L].mean(0); R, _ = orthogonal_procrustes(Y, ref); al[L] = Y @ R * (np.linalg.norm(ref) / max(np.linalg.norm(Y), 1e-8))
    np.savez(O / "fig3a_mds.npz", **{f"aligned_{L}": al[L] for L in LANGS})
    rng, rows, pv = np.random.default_rng(20260927), [], []
    for a, b in PAIRS:
        x = np.concatenate([G[a][s][IU] for s in range(N_STORY)]); y = np.concatenate([G[b][s][IU] for s in range(N_STORY)]); rho = spearmanr(x, y)[0]
        null = [spearmanr(x, np.concatenate([G[b][s][np.ix_(p, p)][IU] for s in range(N_STORY) for p in [rng.permutation(24)]]))[0] for _ in range(1000)]
        rows.append({"pair": f"{a}-{b}", "spearman_rho": rho, "p_perm": (1 + np.sum(np.array(null) >= rho)) / 1001})
        pd.DataFrame({f"{a}_neural_distance": x, f"{b}_neural_distance": y}).to_csv(O / f"fig3e_pairs_{a}_{b}.csv", index=False, float_format="%.5f")
        for L, M in ((a, b), (b, a)):
            for k, sub in enumerate(c[L]):
                r = [spearmanr(S[L][k, s][IU], G[M][s][IU])[0] for s in range(N_STORY)]
                pv.append({"pair": f"{a}-{b}", "language": L, "participant": sub, "similarity": float(np.tanh(np.nanmean(fz(np.array(r)))))})
    pd.DataFrame(rows).to_csv(O / "fig3ce_stats.csv", index=False); pd.DataFrame(pv).to_csv(O / "participant_vs_group.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
