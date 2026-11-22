import argparse
import numpy as np
from langdecoder.surface import parcel_labels
from langdecoder.train import load_decoder
from langdecoder.predict import feeds_for, tr_of, trajectory
from langdecoder.data import load_tokens
from langdecoder.occlusion import region_mask, targets, occlusion_contributions

ap = argparse.ArgumentParser(); ap.add_argument("--decoder", required=True); ap.add_argument("--lang", required=True)
ap.add_argument("--story", default="story09"); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
net, ck = load_decoder(a.decoder, a.device); feeds = feeds_for(ck, "lppc", a.story, "own")
times = load_tokens("lppc", a.lang, a.story, ck["smooth"])[0]; tr = tr_of("lppc")
_, kept = trajectory(net, ck, feeds, times, tr, device=a.device); y = targets(ck, "lppc", a.lang, a.story, kept)
labels = parcel_labels(); parcels = np.array(sorted(set(labels[np.asarray(ck["vox"])].tolist()) - {0}))
base, drop = occlusion_contributions(net, ck, feeds, times, tr, y, {int(p): region_mask(ck["vox"], labels, p) for p in parcels}, device=a.device)
np.savez(a.out, parcels=parcels, drop_cos=np.array([drop[int(p)] for p in parcels]), base_cos=base)
