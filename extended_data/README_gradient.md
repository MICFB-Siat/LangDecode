# Extended Data — semantic geometry gradient, encoding maps, six directions, Neurosynth

Run from the repository root with `export PYTHONPATH=$PWD` (scripts import `generalization/common.py`). Outputs go to `$LANGDECODER_OUT/extended_data/`.

| Figure | Script | Statistics | Output |
|---|---|---|---|
| Three functional gradients (G1–G3) vs shared map | `ed_gradient_three_gradients.py` | Spearman over 400 parcels; two-sided spin (parcel-painted shared map rotated, 1,000, seed 20261010); one-sided test of abs(rho G1) > abs(rho Gk) | `ed_three_gradients.json` |
| ED Fig. 3a, reliability control | `ed_gradient_reliability.py` | split-half parcel reliability per language; partial Spearman on reliability-residualized ranks; one-sided spin (G1 rotated, seed 20261006, reliability fixed) | `ed3a_stats.json`, `ed3a_profile.csv` |
| ED Fig. 3b, shuffled sentence correspondence | `ed_gradient_shuffled_correspondence.py` | 1,000 within-story anchor permutations of the second language (seed 20260930); one-sided P = (k+1)/1,001 | `ed3b_stats.json`, `ed3b_null.csv` |
| ED Fig. 3c, G1-matched comparison | `ed_gradient_g1_matched.py` | higher minus lower half (median split) within G1 thirds; one-sided spin (shared map rotated, seed 20261008); Holm across 3 thirds | `ed3c_stats.json`, `ed3c_parcels.csv` |
| ED Fig. 4, encoding maps | `ed_encoding_maps.py` | LPPC: sign-flip max-T, 10,000 permutations, FWE P < 0.05; Chen2026: cluster mass (t > 2.015), 64 exact sign patterns, cluster FWE P < 0.05 | `ed4_maps.npz` |
| ED Fig. 5, six directions | `ed_directions_scatter.py` | Spearman per direction against the pair-specific map; one-sided spin (pair map rotated, seed 404); P uncorrected, Holm across 6 directions given alongside | `ed5_stats.csv`, `ed5_points.csv` |
| Neurosynth meta-analytic decoding | `ed_neurosynth_maps.py`, `ed_neurosynth_bins.py` | 20 rank bins; two-sided spin test of position difference with BH-FDR over 24 terms; apex enrichment with G1-fit + spun-residual surrogates | `neurosynth_*_bins.csv`, `neurosynth_stats.json` |

Reproduction check (2026-10-15, stored inputs): shared map vs G1 rho = 0.797; per-direction rho 0.367 / 0.256 / 0.245 / 0.159 / 0.096 / 0.242;
ED Fig. 3c differences 0.013 / 0.031 / 0.034 with spin P 0.071 / 0.006 / 0.014 (Holm 0.071 / 0.018 / 0.028), identical to the reported values.
