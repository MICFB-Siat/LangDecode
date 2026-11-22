import numpy as np
from joblib import Parallel, delayed
from _common import (OUT3, FEATURES, LANGS, PAIRS, N_STORY, N_CORTEX, cohort, events, run_files, story_betas, lppc_anchors,
                     neighbourhoods, searchlight_rdms, cosine_rdm_vec, rank_rows, fz, edge_permutation, bh_fdr)

O = OUT3 / "searchlight"; N_EDGE = 276; SEED = 20260908


def semantic_rdms(anchors):
    out = {}
    for L in LANGS:
        ev, f = events(L), np.load(FEATURES / f"lppc_{L}_bloom20.npy"); key = {(s, e): i for i, (s, e) in enumerate(zip(ev.story, ev.event_index_within_story))}
        out[L] = np.stack([cosine_rdm_vec(f[[key[(s, int(k))] for k in anchors[s]]]) for s in range(1, N_STORY + 1)])
    return out


def participant(lang, sub, anchors, sem):
    dest = O / f"{lang}_{sub}.npz"
    if dest.exists(): return
    member, ev_all, files = neighbourhoods(), events(lang), run_files(sub)
    rdm = np.zeros((N_STORY, N_CORTEX, N_EDGE), np.float16); z = np.full((N_STORY, N_CORTEX), np.nan, np.float32); sr = rank_rows(sem[lang])
    for s in range(1, N_STORY + 1):
        ev = ev_all[ev_all.story == s].reset_index(drop=True); b, valid = story_betas(files[s - 1], ev, rows=anchors[s])
        r = searchlight_rdms(b, member, valid); rdm[s - 1] = r; z[s - 1] = fz(np.einsum("ve,e->v", rank_rows(r), sr[s - 1]))
    np.savez(dest, rdm=rdm, semantic_z=np.nanmean(z, 0))


def signflip(maps, n_perm, seed):

    ok = np.isfinite(maps).all(0); x = np.where(ok, maps, 0.0); obs = x.mean(0); rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], np.float32), size=(n_perm, len(x))); signs[0] = 1.0; exceed = np.zeros(x.shape[1]); mx = np.empty(n_perm)
    for a in range(0, n_perm, 128):
        cur = signs[a:a + 128] @ x / len(x); exceed += (cur >= obs - 1e-12).sum(0); mx[a:a + len(cur)] = cur[:, ok].max(1)
    p = (exceed + 1) / (n_perm + 1); fwe = np.maximum((1 + (mx[None] >= obs[:, None]).sum(1)) / (n_perm + 1), p); q = np.full_like(p, np.nan); q[ok] = bh_fdr(p[ok])
    p[~ok] = fwe[~ok] = np.nan; return np.tanh(obs), q, fwe


def label_permutation(ra, rb, n_perm, seed):

    stat = lambda b: np.nanmean(fz(np.einsum("sve,se->sv", ra, b) if b.ndim == 2 else np.einsum("sve,sve->sv", ra, b)), 0)
    rng = np.random.default_rng(seed); obs = stat(rb); ok = np.isfinite(obs); cnt = np.zeros(N_CORTEX); mx = np.empty(n_perm)
    for k in range(n_perm):
        idx = np.stack([edge_permutation(rng.permutation(24)) for _ in range(N_STORY)])
        rp = np.take_along_axis(rb, idx, 1) if rb.ndim == 2 else np.stack([rb[s][:, idx[s]] for s in range(N_STORY)])
        nl = stat(rp); cnt += nl >= obs; mx[k] = np.nanmax(nl[ok])
    fwe = (1 + (mx[None] >= obs[:, None]).sum(1)) / (n_perm + 1); fwe[~ok] = np.nan; return np.tanh(obs), fwe


def main():
    O.mkdir(parents=True, exist_ok=True); c, anc = cohort(), lppc_anchors(); sem = semantic_rdms(anc)
    Parallel(n_jobs=8, verbose=5)(delayed(participant)(L, s, anc, sem) for L in LANGS for s in c[L])
    out, rank = {}, {}
    for L in LANGS:
        tot = np.zeros((N_STORY, N_CORTEX, N_EDGE), np.float32); n = np.zeros(tot.shape, np.uint16); pm = []
        for s in c[L]:
            z = np.load(O / f"{L}_{s}.npz"); r = z["rdm"].astype(np.float32); f = np.isfinite(r); tot += np.where(f, r, 0); n += f; pm.append(z["semantic_z"])
        np.save(O / f"participant_semantic_z_{L}.npy", np.stack(pm).astype(np.float32))
        out[f"participant_semantic_{L}_rho"], out[f"participant_semantic_{L}_q_fdr"], out[f"participant_semantic_{L}_p_fwe"] = signflip(np.stack(pm), 5000, SEED)
        g = np.full(tot.shape, np.nan, np.float32); np.divide(tot, n, out=g, where=n > 0); rank[L] = np.stack([rank_rows(g[s]) for s in range(N_STORY)])
        out[f"semantic_{L}_rho"], out[f"semantic_{L}_p_fwe"] = label_permutation(rank[L], rank_rows(sem[L]), 1000, SEED)
    for a, b in PAIRS:
        out[f"cross_{a}_{b}_rho"], out[f"cross_{a}_{b}_p_fwe"] = label_permutation(rank[a], rank[b], 300, SEED)
    np.savez_compressed(OUT3 / "fig3gi_maps.npz", **out)


if __name__ == "__main__":
    main()
