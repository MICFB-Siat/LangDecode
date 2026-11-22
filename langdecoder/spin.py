import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .paths import SPHERES
from .surface import parcel_mean


class SpinNull:


    def __init__(self, cifti_axis, n_spins=1000, seed=0):
        coords, verts = [], []
        for (_, _, model), hemi in zip(list(cifti_axis.iter_structures())[:2], ("L", "R")):
            xyz = np.asarray(nib.load(str(SPHERES / f"tpl-fsLR_den-32k_hemi-{hemi}_sphere.surf.gii")).darrays[0].data, float)
            coords.append(xyz / np.linalg.norm(xyz, axis=1, keepdims=True)); verts.append(np.asarray(model.vertex, int))
        self.n_full = coords[0].shape[0]; self.lv, self.rv = verts
        self.trees = (cKDTree(coords[0]), cKDTree(coords[1]))
        self.targets = (coords[0][verts[0]], coords[1][verts[1]])
        self.n_spins, self.rng = n_spins, np.random.default_rng(seed)

    def to_full(self, values):
        full = np.full(2 * self.n_full, np.nan)
        full[self.lv] = values[:len(self.lv)]; full[self.n_full + self.rv] = values[len(self.lv):]
        return full

    def rotations(self):
        mirror = np.diag([-1.0, 1.0, 1.0])
        for _ in range(self.n_spins):
            r = Rotation.random(random_state=self.rng).as_matrix()
            li = self.trees[0].query(self.targets[0] @ r.T, k=1)[1]
            ri = self.trees[1].query(self.targets[1] @ (mirror @ r @ mirror).T, k=1)[1]
            yield np.concatenate([li, self.n_full + ri])


def spun_parcel_maps(vertex_map, spin):

    full = spin.to_full(np.where(np.isfinite(vertex_map), vertex_map, np.nanmean(vertex_map))); out = []
    for idx in spin.rotations():
        p = parcel_mean(full[idx]); out.append(np.where(np.isfinite(p), p, np.nanmean(p)))
    return np.asarray(out)


def p_one_sided(observed, null):

    null = np.asarray(null); return (1 + np.sum(null >= observed, axis=0)) / (1 + len(null))
