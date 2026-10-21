# — shared semantic geometry gradient and cross-language generalization

Run from the repository root with `export PYTHONPATH=$PWD` and `LANGDECODER_DATA` / `LANGDECODER_OUT` set (see `langdecoder/paths.py`).
Scripts read intermediate artefacts listed in `generalization/common.py` (group searchlight RDMs from `semantic_geometry/`, trained decoders from `langdecoder/`).

| Panel | Script | Command | Output (`$LANGDECODER_OUT/generalization/`) |
|---|---|---|---|
| Shared semantic geometry gradient (input to all grouping) | `shared_map.py` | `python generalization/shared_map.py` | `shared_map_train_{pair}.npy`, `shared_map_parcels.csv` |
| Gradient maps, Low/Mid/High maps and contours (d, e, g), correspondence with G1 (rho = 0.80) | `fig4_gradient_correspondence.py` | `python generalization/fig4_gradient_correspondence.py` | `gradient_correspondence.{json,csv}`, `gradient_tiers_vertex.npz` (rendered with Connectome Workbench) |
| c, NEBULA101 zero-shot identification | `fig4c_nebula_zeroshot.py`, `fig4c_nebula_stats.py` | `python generalization/fig4c_nebula_zeroshot.py`; `... --untrained 0/1/2`; `python generalization/fig4c_nebula_stats.py` | `fig4c_prism.csv`, `nebula_stats.json` |
| NEBULA101 Low/Mid/High inputs | same scripts | (written by the trained run) | `nebula_tiers_{intact,foreign}.csv` |
| Single-region cross-language decoding (scatter; input to ED Fig. 3 and 5) | `fig4_single_region_2afc.py` | `python generalization/fig4_single_region_2afc.py cn2en` (six directions) | `single_region_{dir}.npz` |
| Tier-restricted and tier-lesioned text decoding (LPPC, Chen2026) | `fig4_tier_decoding.py`, `fig4_tier_stats.py` | `python generalization/fig4_tier_decoding.py lppc cn2en`; `... chen en2zh`; generate and score texts with `langdecoder` (beam search, BGEScore 20-s windows) into `$LANGDECODER_OUT/texts/score_{tag}.json`; `python generalization/fig4_tier_stats.py` | `traj/*.npz`, `lppc_lesion_drop.csv`, `chen_tier_bge.csv`, `tier_decoding_stats.json` |
| f, cross-language neural response prediction by group | `encoding.py`, `fig4f_encoding.py` | `python generalization/fig4f_encoding.py fit CN CN001` (per participant); `python generalization/fig4f_encoding.py summary` | `fig4f_prism.csv`, `fig4f_stats.json` |

Library modules: `common.py` (inputs, thirds), `crosslang.py` (pathway-averaged prediction, keep-only inputs, 20-s segment 2AFC), `encoding.py` (LSS and ridge encoding).

Statistics as in the figures: gradient correspondence one-sided spin (G1 rotated, 1,000, seed 20261008); NEBULA two-sided paired Wilcoxon vs untrained (French n = 97, English n = 79) and one-sided paired Wilcoxon between tiers; lesion exact sign-flip over 6 directions (64 patterns, one-sided); Chen2026 exact sign-flip over 6 participants; Fig. 4f paired Wilcoxon with Holm over 3 comparisons (one- and two-sided reported).

Notes on the reported runs
- Decoders are those of the reported results (`langdecoder/README.md` documents their training chain).
- Text generation for the tier and lesion panels used the training-story lexicon restriction in the beam (`--mask lexicon`); keep this option to reproduce the reported numbers.
- Fig. 4f groups were defined by thirds of the parcel-mean training-story map averaged over CN–EN and EN–FR (as in the reported figure); NEBULA groups use the nine-story three-pair map; all other groups use the training-story maps.
- Single-region and lesion analyses use pair-specific thirds (parcels with >= 10 vertices); random thirds use the seeds in `fig4_tier_decoding.py`.
