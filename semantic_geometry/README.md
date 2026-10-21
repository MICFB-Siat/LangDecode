# — shared cortical semantic geometry

Run from the repository root with `PYTHONPATH=.` (scripts import `langdecoder` and `semantic_geometry/_common.py`).
Data locations are set in `langdecoder/paths.py` (`LANGDECODER_DATA`, `LANGDECODER_OUT`); Fig. 3 additionally expects
`DATA/events/` (sentence tables per language, 10-s window tables), `DATA/features/` (BLOOM-1b1 layer-20 sentence vectors),
`DATA/norms/` (word norms), `DATA/atlas/searchlight_8mm_csr.npz` (8-mm geodesic searchlight membership) and
`DATA/LPPC-fMRI/participants.json` (cohort per language).

| Panel | Script | Command | Main output (`OUT/semantic_geometry/`) |
|---|---|---|---|
| (shared) sentence LSS betas | `lss_betas.py` | `python semantic_geometry/lss_betas.py` and `--phase-null` | `betas/`, `betas_prnull/` |
| 3a MDS, 3e event-pair distances | `fig3ace_group_rdms.py` | `python semantic_geometry/fig3ace_group_rdms.py` | `anchors/fig3a_mds.npz`, `anchors/fig3e_pairs_*.csv`, `anchors/fig3ce_stats.csv` |
| 3b relational alignment | `fig3b_relational_alignment.py` | `python semantic_geometry/fig3b_relational_alignment.py` | `relational_alignment/fig3b_prism.csv`, `relational_alignment/fig3b_stats.csv` |
| 3c semantic attribute arrows (3-D) | `fig3c_attribute_arrows.py` | `python semantic_geometry/fig3c_attribute_arrows.py` | `semantic_attributes/fig3c_arrows.csv`, `semantic_attributes/fig3c_angles.csv`, `semantic_attributes/fig3c_arrows.pdf` |
| 3d semantic attribute directions | `fig3d_semantic_attributes.py` | `python semantic_geometry/fig3d_semantic_attributes.py` | `semantic_attributes/fig3d_prism.csv`, `semantic_attributes/fig3d_stats.csv` |
| 3g brain–semantic RSA, 3i cross-language RSA | `fig3gi_searchlight_rsa.py` | `python semantic_geometry/fig3gi_searchlight_rsa.py` | `fig3gi_maps.npz`, `searchlight/participant_semantic_z_*.npy` |
| 3h, 3j individual correspondence | `fig3hj_individual_correspondence.py` | `python semantic_geometry/fig3hj_individual_correspondence.py` | `individual_correspondence/Fig3h_prism.csv`, `individual_correspondence/Fig3j_prism.csv`, `individual_correspondence/fig3hj_stats.csv` |
| Chen2026 replication (3e panel, supplement) | `fig3_chen2026.py` | `python semantic_geometry/fig3_chen2026.py` | `chen2026/*.csv` |
| Extended Data: stability across stories and narrative time | `ed_stability.py` | `python semantic_geometry/ed_stability.py` | `stability/story_split_stability.csv`, `stability/narrative_windows_binned.csv` |

Order: `lss_betas.py` (both modes) → `fig3gi_searchlight_rsa.py` → the remaining scripts.

## Statistics
- 3b: one-sided Wilcoxon (> 0) per language pair; each point is a target-language participant.
- 3e: Spearman across event pairs pooled over stories; 1,000 within-story anchor-label permutations.
- 3c: descriptive; pairwise angles between language arrows in the full cortical space.
- 3d: one-sided paired Wilcoxon (parallelism vs the listener's random-regrouping control), n = 111 pooled over languages, Holm over 3 attributes.
- 3g/3i: participant maps — 5,000 one-sided sign flips (BH-FDR, max-statistic FWE); group maps — 1,000 (brain–semantic) and 300 (cross-language) anchor-label permutations, max-statistic FWE.
- 3h/3j: one-sided Wilcoxon of (ρ − listener's spin-null median) per language, Holm over 3 languages.

## Reproduction checks
- 3b metric (relational alignment and circular-diagonal excess): identical to the original implementation on test data (max difference 0).
- 3h/3j: the repo code path reproduces the stored values for a test listener exactly (ρ −0.06805 / −0.09099; spin-null medians 0.00106 / −0.00513).

## Known deviations and notes
- **3h/3j spin null:** the published values used the first 300 of 1,000 rotations (seed 20261004); the script keeps this (`N_ROT = 300`).
- **Preprocessing:** representational analyses (3g–3j and 3b) used the DCAN-processed CIFTI (0.009–0.08 Hz band-pass with nuisance regression); decoding analyses used the series without this band-pass. Point `run_files(..., root=...)` at the corresponding derivatives.
- **3j shared map:** the published panel used the group cross-language maps from the September run of the same searchlight pipeline; the script recomputes them from `fig3gi_searchlight_rsa.py`.
- **3g display maps:** the published 3g maps came from an exploratory 6-sentence-per-story anchor subset whose selection used the other cohorts' neural data. The script uses the 24 evenly spaced anchors (outcome-independent), which is the version behind the 3h/3j statistics.
- **3b vs Methods:** the published 3b is the relational-alignment analysis (20 story-position bins, Destrieux-148 parcels). The Methods text currently describes participant-RDM-to-group-RDM similarity (24 anchors); that variant is written by `fig3ace_group_rdms.py` (`anchors/participant_vs_group.csv`).
- **3b inputs:** the 10-s window table and its window responses were cached by an earlier pipeline; `fig3b_relational_alignment.py` re-implements the window extraction (z-scored run, parcel mean, mean over the window's TRs).
- **Design null:** phase randomisation uses a deterministic per-run seed (CRC32 of language/participant/story); the original used Python `hash`, which is not reproducible across sessions, so the null differs by random draw only.
- **EN084:** this participant failed the DCAN step; for representational analyses his non-band-passed series was band-passed (0.009–0.08 Hz, zero-phase Butterworth) without DCAN nuisance regression.
