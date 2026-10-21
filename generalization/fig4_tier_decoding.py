import sys, numpy as np, pandas as pd
from langdecoder.paths import OUT
from langdecoder.surface import parcel_labels, parcel_mean
from common import shared_map_vertex, pair_of, thirds
import crosslang as X


def lppc_sets(direction):
    par = parcel_labels(); ids = np.load(OUT / "generalization" / f"single_region_{direction}.npz")["parcel_ids"]
    geo = parcel_mean(shared_map_vertex(pair_of(direction)))[ids - 1]
    tier = pd.qcut(pd.Series(geo).rank(method="first"), 3, labels=False).to_numpy()
    sets = {f"keep_{n}": np.isin(par, ids[tier == k]) for k, n in enumerate(("Low", "Mid", "High"))}
    rng = np.random.default_rng(20261004 + sum(map(ord, direction))); rand = [set(rng.choice(ids, len(ids) // 3, replace=False).tolist())]
    for k in (2, 3): rand.append(set(np.random.default_rng(20261011 + 97 * k + sum(map(ord, direction))).choice(np.arange(1, 401), 133, replace=False).tolist()))
    sets["keep_Random"] = np.isin(par, list(rand[0])); sets["whole"] = np.isin(par, ids)
    for k, n in enumerate(("Low", "Mid", "High")): sets[f"minus_{n}"] = np.isin(par, ids[tier != k])
    for k, r in enumerate(rand, 1): sets[f"minus_Random{k}"] = (par > 0) & ~np.isin(par, list(r))
    return sets


def chen_sets():
    par = parcel_labels(); t = thirds(parcel_mean(shared_map_vertex()))
    sets = {f"keep_{n}": np.isin(par, np.flatnonzero(t == k) + 1) for k, n in enumerate(("Low", "Mid", "High"))}
    sets["keep_Random"] = np.isin(par, np.random.default_rng(0).choice(400, 133, replace=False) + 1); return sets


def main(dataset, direction):
    a, b = direction.split("2"); (OUT / "generalization" / "traj").mkdir(parents=True, exist_ok=True)
    if dataset == "lppc":
        net, norm = X.load_decoder(f"wc_{a}"); feeds = X.group_feed(open(OUT / f"subjects_{b}.txt").read().split()); story, sets = "story09", lppc_sets(direction)
        n_rows = len(feeds)
    else:
        net, norm = X.load_decoder(f"chen_{a}"); subs = open(OUT / "subjects_chen.txt").read().split(); story, sets = "story10", chen_sets()
        feeds = {i: X.zscore(np.load(OUT / "bold_chen" / s / f"bold_{b}_{story}.npy")) for i, s in enumerate(subs)}; n_rows = min(len(v) for v in feeds.values())
    times, y = X.targets(f"chen_{b}" if dataset == "chen" else b, norm, story); lo, w, keep = X.delay_index(n_rows, times)
    for name, kv in sets.items():
        if dataset == "lppc":
            p = X.predict(net, norm, feeds, lo, w, kv)
            np.savez(OUT / "generalization" / "traj" / f"{dataset}_{direction}_{name}.npz", pred=p.cpu().numpy(), times=times[keep], cosine=X.cosine(p, y[keep]))
        else:
            for i in feeds:
                p = X.predict(net, norm, feeds, lo, w, kv, pathways=[i])
                np.savez(OUT / "generalization" / "traj" / f"{dataset}_{direction}_{name}_{i + 1}.npz", pred=p.cpu().numpy(), times=times[keep])
        print(direction, name, int(kv.sum()), "vertices")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
