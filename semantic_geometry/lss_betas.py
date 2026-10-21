import argparse, zlib
import numpy as np
from joblib import Parallel, delayed
from _common import OUT3, LANGS, N_STORY, cohort, events, run_files, story_betas


def subject(lang, sub, out, null):
    dest = out / f"{lang}_{sub}.npy"
    if dest.exists(): return
    ev_all, files, parts = events(lang), run_files(sub), []
    for s in range(1, N_STORY + 1):
        ev = ev_all[ev_all.story == s].sort_values("event_index_within_story").reset_index(drop=True)
        seed = zlib.crc32(f"{lang}_{sub}_{s}".encode()) if null else None
        parts.append(story_betas(files[s - 1], ev, phase_seed=seed)[0])
    np.save(dest, np.concatenate(parts).astype(np.float16))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase-null", action="store_true"); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); out = OUT3 / ("betas_prnull" if a.phase_null else "betas"); out.mkdir(parents=True, exist_ok=True); c = cohort()
    Parallel(n_jobs=a.jobs, verbose=5)(delayed(subject)(L, s, out, a.phase_null) for L in LANGS for s in c[L])
    for L in LANGS:
        ev = events(L).sort_values(["story", "event_index_within_story"]).reset_index(drop=True)
        g = np.mean([np.load(out / f"{L}_{s}.npy").astype(np.float32) for s in c[L]], 0)
        np.savez(out / f"group_{L}.npz", beta=g.astype(np.float16), story=ev.story.to_numpy(), sentence_index=ev.sentence_index.to_numpy(),
                 n_words=ev.n_words.to_numpy(), dur=(ev.offset - ev.onset).to_numpy(), text=ev.text.astype(str).to_numpy(), n_subjects=len(c[L]))


if __name__ == "__main__":
    main()
