import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import chi2, norm
from nilearn.datasets import load_mni152_brain_mask
from nilearn import surface
import nibabel as nib
from langdecoder.paths import DATA, OUT, SPHERES, ATLAS

TERMS = ["motor", "visual perception", "auditory", "somatosensory", "eye movements", "multisensory", "action", "pain", "reward", "attention",
         "working memory", "cognitive control", "inhibition", "emotion", "language", "semantic", "reading", "autobiographical memory",
         "theory mind", "self referential", "episodic memory", "numerical", "face", "speech"]
TAL2MNI = np.linalg.inv(np.array([[0.9254, 0.0024, -0.0118, -1.0207], [-0.0048, 0.9316, -0.0871, -1.7667], [0.0152, 0.0883, 0.8924, 4.0926], [0, 0, 0, 1]]))


def main():
    R = DATA / "neurosynth"; c = pd.read_csv(R / "data-neurosynth_version-7_coordinates.tsv.gz", sep="\t"); m = pd.read_csv(R / "data-neurosynth_version-7_metadata.tsv.gz", sep="\t")
    f = np.load(R / "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz", allow_pickle=True)
    F = sp.csr_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"])).tocsc(); vocab = [t for t in open(R / "data-neurosynth_version-7_vocab-terms_vocabulary.txt").read().split("\n") if t]
    ids = m.id.astype(str).tolist(); space = dict(zip(ids, m.space.astype(str)))
    mask = load_mni152_brain_mask(resolution=2); A, inv, M = mask.affine, np.linalg.inv(mask.affine), mask.get_fdata() > 0; shp = M.shape
    g = np.mgrid[-3:4, -3:4, -3:4].reshape(3, -1).T; ball = g[np.linalg.norm(g * 2.0, axis=1) <= 6.0]; act = {}
    for sid, grp in c.assign(id=c.id.astype(str)).groupby("id"):
        xyz = grp[["x", "y", "z"]].to_numpy(float)
        if space.get(sid, "MNI").upper().startswith("TAL"): xyz = (TAL2MNI @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
        v = (np.rint((inv @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]).astype(int)[:, None] + ball[None]).reshape(-1, 3)
        act[sid] = np.unique(np.ravel_multi_index(v[np.all((v >= 0) & (v < shp), 1)].T, shp))
    studies = [s for s in ids if s in act]; row = {s: i for i, s in enumerate(ids)}; N = len(studies); total = np.zeros(np.prod(shp))
    for s in studies: total[act[s]] += 1
    at = np.load(ATLAS, allow_pickle=True); idx, par = at["idx64984"], at["parcel"]; P = []
    for w in TERMS:
        col = F[:, vocab.index(w)].toarray().ravel(); term = [s for s in studies if col[row[s]] > 0.001]; nT = len(term); a = np.zeros_like(total)
        for s in term: a[act[s]] += 1
        b, cc = nT - a, total - a; d = (N - nT) - cc; den = (a + b) * (cc + d) * (a + cc) * (b + d)
        X2 = np.where(den > 0, N * (a * d - b * cc) ** 2 / np.maximum(den, 1e-12), 0)
        z = norm.isf(np.clip(chi2.sf(X2, 1), 1e-300, 1) / 2) * np.sign(a / max(nT, 1) - cc / max(N - nT, 1)); z[~M.ravel()] = 0
        img = nib.Nifti1Image(z.reshape(shp).astype(np.float32), A)
        v = np.concatenate([surface.vol_to_surf(img, str(SPHERES / f"tpl-fsLR_den-32k_hemi-{h}_midthickness.surf.gii"), radius=3.0, interpolation="linear") for h in "LR"])[idx]
        P.append([np.nanmean(v[par == p]) for p in range(1, 401)]); print(w, nT)
    np.savez(OUT / "extended_data" / "neurosynth_term_parcels.npz", terms=np.array(TERMS), parcels=np.array(P))


if __name__ == "__main__":
    main()
