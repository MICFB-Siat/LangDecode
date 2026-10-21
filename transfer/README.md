# — cross-language semantic transfer with limited target-language data

Run from this directory with the repository root on `PYTHONPATH` and `LANGDECODER_DATA` / `LANGDECODER_OUT` set.

| Panel | Script | Command | Output (under `$LANGDECODER_OUT`) |
|---|---|---|---|
| a–c (networks, texts) | `run_transfer.sh` (uses `persub_beam.py`) | `bash run_transfer.sh` | `transfer/{lppc,chen}_*_p{30,50,70}_{base,xl}.pt`, `transfer/text/score_*_sub-*.json` |
| a–c (data for Prism) | `a_c_transfer_prism.py` | `python a_c_transfer_prism.py` | `transfer/prism_fig5_{dataset}_{lang}.csv`, `transfer/fig5ac_summary.csv` |
| d (contribution map) | `occlusion_30pct.py` (called by `run_transfer.sh`), `d_e_contribution.py` | `python d_e_contribution.py` | `transfer/fig5d_contribution_percentile_parcels.npy` |
| e (contribution by group) | `d_e_contribution.py` | same | `transfer/prism_fig5e_contribution_by_group.csv`, `transfer/fig5e_spin_tests.json` |
| f (groups) | `nebula_tier_masks.py` | `python nebula_tier_masks.py` | `nebula/parcel_tertiles_geo.npz` |
| f (training) | `f_nebula_fewshot.py` | `for k in T1 T2 T3 R1; do for i in smn cn fr; do for f in 0 1; do python f_nebula_fewshot.py --keep $k --init $i --frac 0.3 --fold $f; done; done; done` | `nebula/fewshot_tier/*_pred.npz` |
| f (data for Prism) | `f_nebula_tiers.py` | `python f_nebula_tiers.py` | `nebula/fewshot_tier/prism_fig5f_langdecoder_ChinesePretrained_30pct.csv` |

Design. 30/50/70 % = the first 2/4/6 LPPC-fMRI training stories (validation story 8, test story 9) or the first 3/4/6 Chen2026 stories
(validation 9, test 10). Baseline starts from the SMN4Lang trunk; langdecoder first pretrains that trunk on the source listeners
(LPPC-fMRI: the two other languages; Chen2026: the same bilinguals' other-language runs) and is then fitted to the same target subset.

Inputs produced elsewhere in the repository: `models/trunk_smn.pt` and `models/wc_{cn,fr}.pt` (langdecoder), the training-story maps
`rsa/xrsa_train_{CN_EN,EN_FR}.npy` and the nine-story maps `rsa/xrsa_all_{CN_EN,CN_FR,EN_FR}.npy` (cross-language searchlight RSA),
and the NEBULA101 time series, normalisation packs and passage targets `nebula/{ts/,norm_{cn,fr}.npz,zeroshot_{cn,fr}_pred.npz,zeroshot_cn.csv}`
(NEBULA101 zero-shot scripts, Fig. 4c).

Statistics.
- a–c: GraphPad Prism, two-way ANOVA (model as repeated-measures factor, data proportion between participants), Šídák comparisons of
  Baseline vs langdecoder within each proportion.
- e: one-sided spin tests (1,000 bilaterally coupled rotations; groups recomputed on every rotation), Mid − Low and High − Mid, Holm over 2.
- f: GraphPad Prism, one-way repeated-measures ANOVA (Geisser–Greenhouse) with Tukey's multiple comparisons.

Reproduction checks (archived inputs): `d_e_contribution.py` gives spin P = 0.007 (Mid − Low) and 0.001 (High − Mid), Holm 0.007 / 0.002;
`f_nebula_tiers.py` gives the Fig. 5f means 0.5179 / 0.5631 / 0.6032 (Low / Mid / High, n = 97).

Notes.
- The per-listener values plotted in the paper's Fig. 5a–c were supplied from the authors' own runs of this design; `a_c_transfer_prism.py`
  produces the same table layout from `run_transfer.sh`.
- Fig. 5f uses the Chinese-pretrained decoder only; `prism_nebula_tier_accuracy_all.csv` also gives the mean of the Chinese- and
  French-pretrained decoders.
- `MASK=lexicon` (default in `run_transfer.sh`) reproduces the archived texts, whose beam candidates were restricted to the training-story lexicon.
