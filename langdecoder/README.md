# langdecoder

Multi-subject mapping from cortical fMRI responses to the BLOOM-1b1 layer-20 semantic space, and text reconstruction by semantic consistency-guided beam search.

| Module | Content |
|---|---|
| `model.py` | `Net2`: subject-specific projection and feature-wise scaling/shifting, shared trunk (dilated temporal convolutions, Transformer, attention pooling over delays), projection head; loss (symmetric InfoNCE with learnable temperature + 1 − cosine); stage-to-stage initialization |
| `data.py` | TextGrid reading, BPE token timing, BLOOM-1b1 layer-20 targets, run z-scoring, responses at delays 0–12 s by linear interpolation, per-subject GPU store, training-rate clock |
| `train.py` | `pretrain` and `finetune` entry points; `load_decoder` |
| `predict.py` | Test-time semantic trajectories (own runs, or a target-language group mean through every source pathway) |
| `beam.py` | Semantic consistency-guided beam search (k = 12, M = 96, 15-token context) |
| `score.py` | BGEScore, WER, BLEU-1, METEOR in 20-s windows; temporal nulls |
| `occlusion.py` | Keep-only, lesion and occlusion evaluation of trained decoders; 20-s segment identification |
| `paths.py`, `surface.py`, `spin.py`, `stats.py`, `rsa.py` | Data paths, cortical surface helpers, spin permutations, statistics and RSA |
| `random_baseline.py` | Random baseline: the same beam search with candidate scores drawn from U(0, 1) |

Data layout and paths: see `data.py` and `langdecoder/paths.py` (`LANGDECODER_DATA`, `LANGDECODER_OUT`).

## Commands

```bash
python -c "from langdecoder.data import cifti_to_run; cifti_to_run('sub-EN057_run-1_Atlas.dtseries.nii', 'data/lppc/bold/EN057/story01.npy')"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer; from langdecoder.data import build_tokens; from langdecoder.paths import BLOOM
m = AutoModelForCausalLM.from_pretrained(str(BLOOM)).cuda().eval(); t = AutoTokenizer.from_pretrained(str(BLOOM))
build_tokens('transcripts/lppc/en/story01.TextGrid', 'data/lppc/tokens/en/story01.npz', m, t)"

python -m langdecoder.train pretrain --sources lppc:cn,lppc:en,lppc:fr,chen:en --vox-file pretrain_vertices.npy --out trunk_pre.pt
python -m langdecoder.train finetune --dataset smn4lang --lang zh --fit 1-51 --val 52-57 --test 58-60 \
    --smooth 4 --batch 512 --vox-file pretrain_vertices.npy --init trunk_pre.pt --out trunk_smn.pt

for L in cn en fr; do python -m langdecoder.train finetune --dataset lppc --lang $L --init trunk_smn.pt --out lppc_$L.pt; done

python -m langdecoder.beam --decoder lppc_en.pt --dataset lppc --lang en --feed own \
    --train-grids "transcripts/lppc/en/story0[1-8].TextGrid" --out gen_en.json

python -m langdecoder.score --generation gen_en.json --transcript transcripts/lppc/en/story09.TextGrid --language en --out score_en.json

for s in $(seq 0 199); do python -m langdecoder.random_baseline --seed $s --decoder lppc_en.pt --dataset lppc --lang en --feed own \
    --train-grids "transcripts/lppc/en/story0[1-8].TextGrid" --out random/en_$s.json; done
```

Hyperparameters (defaults): d = 256, 3 dilated convolution blocks, 2 Transformer layers (4 heads), dropout 0.3, AdamW with weight decay 0.1 and one-cycle schedule; pretraining 10 epochs, batch 512, learning rate 3 × 10⁻⁴, input noise 0.1; subsequent stages 20 epochs, learning rate 1 × 10⁻⁴, input noise 0.05, best validation epoch kept.

**Note.** Pretraining here is the supervised trunk pretraining that produced the reported decoders (InfoNCE + cosine to BLOOM-1b1 layer-20 targets on the source datasets); only the subject-independent trunk is transferred, and subject-specific projections and scaling/shifting are newly initialized for each dataset.

## Decoding options

| Flag | Default | Alternative | Used for the reported texts |
|---|---|---|---|
| `--clock` | `rate`: constant clock from the training-story speech rate | `tokens`: token times of the test transcript | LPPC whole-cortex texts: `tokens` |
| `--mask` | `none`: full PMM vocabulary | `lexicon`: tokens that occur in the training transcripts | all reported texts: `lexicon` |

Use the right-hand settings to reproduce the archived results.
