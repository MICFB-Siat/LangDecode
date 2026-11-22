import sys, numpy as np
from langdecoder.paths import OUT
from langdecoder.surface import parcel_labels
import crosslang as X


def main(direction):
    a, b = direction.split("2"); net, norm = X.load_decoder(f"wc_{a}")
    feed = X.group_feed(open(OUT / f"subjects_{b}.txt").read().split()); times, y = X.targets(b, norm)
    lo, w, keep = X.delay_index(len(feed), times); times, y = times[keep], y[keep]
    full = X.two_afc(X.predict(net, norm, feed, lo, w), y, times); par = parcel_labels(); ids, acc = [], []
    for q in range(1, 401):
        kv = par == q
        if kv.sum() < 10: continue
        ids.append(q); acc.append(X.two_afc(X.predict(net, norm, feed, lo, w, kv), y, times))
    np.savez(OUT / "generalization" / f"single_region_{direction}.npz", parcel_ids=np.array(ids), parcel_2afc=np.array(acc), full_2afc=full)
    print(direction, "whole cortex 2AFC %.3f, median parcel %.3f" % (full, np.median(acc)))


if __name__ == "__main__":
    main(sys.argv[1])
