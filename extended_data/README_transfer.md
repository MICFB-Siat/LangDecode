# Extended Data — transfer analyses (prefix `ed_transfer_`)

| Panel | Script | Command | Output (under `$LANGDECODER_OUT`) |
|---|---|---|---|
| NEBULA101 transfer gain by semantic geometry consistency group at 30/50/70 % of listeners (optional) | `ed_transfer_nebula_fraction_gain.py` | run `transfer/f_nebula_fewshot.py` for `--frac 0.3 0.5 0.7`, then `transfer/f_nebula_tiers.py`, then `python ed_transfer_nebula_fraction_gain.py` | `nebula/fewshot_tier/ed_transfer_gain_summary.csv`, `prism_ed_transfer_gain.csv` |

Gain = (pretrained and fitted LPPC decoder) − (NEBULA-only baseline) per listener; two-sided Wilcoxon against zero per group and for High − Low.
Not included: Chen2026 participant-pair design, size-matched data-source experiment and NEBULA decoder pooling (draft panels; add them if they
enter the Extended Data).
