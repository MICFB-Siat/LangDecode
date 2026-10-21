"""Fig. 3c: concreteness, socialness and arousal directions in the aligned cortical representational spaces.

Group sentence betas of the three languages are standardised and adjusted for word count and log duration jointly across languages
within each story; low- and high-third centroids (thirds as in Fig. 3d) give one arrow per language and attribute. Arrows are shown in
a 3-D frame: Dim 1 = mean arrow direction, Dims 2-3 = leading between-language offsets of the pole centroids orthogonal to Dim 1;
Dims 2-3 of socialness and arousal are rotated (orthogonal Procrustes) onto the concreteness frame. Pairwise angles are computed in
the full cortical space. Outputs: OUT/semantic_geometry/semantic_attributes/fig3c_arrows.csv, fig3c_angles.csv, fig3c_arrows.pdf."""
import numpy as np, pandas as pd
from fig3d_semantic_attributes import O, B, SHOW, LANGS, norms, scores, resid, thirds


def pooled(beta, story, nw, dur):
    """Joint z-scoring across languages within each story, then removal of word count and log duration."""
    X = {L: beta[L].astype(np.float32).copy() for L in LANGS}
    for s in np.unique(story):
        i = story == s; n = int(i.sum()); P = np.concatenate([X[L][i] for L in LANGS]); P = (P - P.mean(0)) / (P.std(0) + 1e-6)
        zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)
        Z = np.concatenate([np.column_stack([np.ones(n), zs(nw[L][i].astype(float)), np.log(dur[L][i] + 1e-3)]) for L in LANGS])
        Zc = Z - Z.mean(0); P = P - Zc @ np.linalg.lstsq(Zc, P - P.mean(0), rcond=None)[0]
        for k, L in enumerate(LANGS): X[L][i] = P[k * n:(k + 1) * n]
    return X


def frame(C):
    """C: [language, pole, vertex] -> pole coordinates [language, pole, 3] in the Dim 1-3 frame."""
    u = (C[:, 1] - C[:, 0]).sum(0); u /= np.linalg.norm(u)
    Y = C.reshape(6, -1) - C.reshape(6, -1).mean(0); R = Y - np.outer(Y @ u, u)
    w = np.linalg.svd(R, full_matrices=False)[2][:2]
    return (Y @ np.stack([u, *w]).T).reshape(3, 2, 3)


def main():
    g = {L: np.load(B / f"group_{L}.npz", allow_pickle=True) for L in LANGS}
    common = set.intersection(*(set(g[L]["sentence_index"]) for L in LANGS)); rows = {L: np.flatnonzero(np.isin(g[L]["sentence_index"], list(common))) for L in LANGS}
    story = g["EN"]["story"][rows["EN"]]
    nw = {L: (np.array([len(t.replace(" ", "")) for t in g[L]["text"]]) if L == "CN" else g[L]["n_words"])[rows[L]] for L in LANGS}
    dur = {L: g[L]["dur"][rows[L]].astype(float) for L in LANGS}
    X = pooled({L: g[L]["beta"][rows[L]] for L in LANGS}, story, nw, dur)
    nrm = norms(); arrows, angles, ref = [], [], None
    for a in SHOW:
        lab = thirds(resid(scores(g["EN"]["text"][rows["EN"]], nrm[a]), story, nw["EN"], dur["EN"]), story)
        C = np.stack([np.stack([X[L][lab == -1].mean(0), X[L][lab == 1].mean(0)]) for L in LANGS]).astype(np.float64)
        P = frame(C)
        if ref is None: ref = P.reshape(6, 3)[:, 1:].copy()
        else:
            U, _, Vt = np.linalg.svd(P.reshape(6, 3)[:, 1:].T @ ref); P = np.concatenate([P.reshape(6, 3)[:, :1], P.reshape(6, 3)[:, 1:] @ (U @ Vt)], 1).reshape(3, 2, 3)
        for k, L in enumerate(LANGS):
            for j, pole in enumerate(("low", "high")): arrows.append({"attribute": a, "language": L, "pole": pole, "dim1": P[k, j, 0], "dim2": P[k, j, 1], "dim3": P[k, j, 2]})
        A = C[:, 1] - C[:, 0]; A /= np.linalg.norm(A, axis=1, keepdims=True)
        for i in range(3):
            for j in range(i + 1, 3): angles.append({"attribute": a, "pair": f"{LANGS[i]}-{LANGS[j]}", "angle_deg": float(np.degrees(np.arccos(np.clip(A[i] @ A[j], -1, 1))))})
    O.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(arrows).to_csv(O / "fig3c_arrows.csv", index=False, float_format="%.5f"); pd.DataFrame(angles).to_csv(O / "fig3c_angles.csv", index=False, float_format="%.2f")
    plot(pd.DataFrame(arrows))


def plot(d):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"CN": "#D55E00", "EN": "#0072B2", "FR": "#009E73"}; fig = plt.figure(figsize=(7.2, 2.6))
    for k, a in enumerate(SHOW):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d"); ax.view_init(elev=20, azim=-62); ax.set_proj_type("ortho")
        for L in LANGS:
            p = d[(d.attribute == a) & (d.language == L)].set_index("pole")[["dim1", "dim2", "dim3"]]
            lo, hi = p.loc["low"].to_numpy(), p.loc["high"].to_numpy(); ax.quiver(*lo, *(hi - lo), color=col[L], lw=1.5, arrow_length_ratio=0.12)
            ax.scatter(*lo, s=14, facecolors="white", edgecolors=col[L]); ax.scatter(*hi, s=14, color=col[L])
        ax.set_title(a.capitalize(), fontsize=8); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlabel("Dim 1", fontsize=6); ax.set_ylabel("Dim 2", fontsize=6); ax.set_zlabel("Dim 3", fontsize=6)
    fig.savefig(O / "fig3c_arrows.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
