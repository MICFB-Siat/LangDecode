import argparse
import langdecoder.beam as beam
from langdecoder.predict import zscore
from langdecoder.data import load_run
from langdecoder.train import load_decoder

ap = argparse.ArgumentParser(); ap.add_argument("--decoder", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--lang", required=True)
ap.add_argument("--story", required=True); ap.add_argument("--train-grids", required=True); ap.add_argument("--mask", default="lexicon")
ap.add_argument("--out-dir", required=True); ap.add_argument("--name", required=True)
a = ap.parse_args()
subjects = load_decoder(a.decoder)[1]["subjects"]
for i, s in enumerate(subjects):
    beam.feeds_for = lambda ck, dataset, story, mode, target_subjects=None, i=i, s=s: {i: zscore(load_run(dataset, s, story, ck["vox"]))}
    beam.main(["--decoder", a.decoder, "--dataset", a.dataset, "--lang", a.lang, "--story", a.story, "--feed", "own", "--train-grids", a.train_grids,
               "--mask", a.mask, "--out", f"{a.out_dir}/gen_{a.name}_sub-{s}.json"])
