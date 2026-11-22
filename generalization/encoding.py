import numpy as np
from scipy.stats import gamma
from sklearn.decomposition import PCA

ALPHAS = [10.0 ** k for k in range(-2, 7)]


def spm_hrf(tr=2.0, oversampling=20, duration=32.0):
    t = np.arange(0, duration, tr / oversampling); h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0; return h / h.sum()


def lss_weights(onsets, offsets, n_tr, tr=2.0, oversampling=20, high_pass=128.0):

    dt, nh = tr / oversampling, n_tr * oversampling; X = np.zeros((nh, len(onsets)))
    for j, (a, b) in enumerate(zip(onsets, offsets)):
        s = max(0, int(np.floor(a / dt))); X[s:min(nh, max(s + 1, int(np.ceil(b / dt)))), j] = 1.0
    h = spm_hrf(tr, oversampling); X = np.stack([np.convolve(X[:, j], h)[:nh] for j in range(X.shape[1])], 1)[np.arange(n_tr) * oversampling]
    X /= np.maximum(np.abs(X).max(0, keepdims=True), 1e-12); t = np.arange(n_tr)
    dct = np.column_stack([np.ones(n_tr)] + [np.cos(np.pi * (t + .5) * k / n_tr) for k in range(1, int(2 * n_tr * tr / high_pass) + 1)])
    total = X.sum(1); return np.stack([np.linalg.pinv(np.column_stack([X[:, j], total - X[:, j], dct]), rcond=1e-8)[0] for j in range(X.shape[1])]).astype(np.float32)


def zscore(x):
    x = np.asarray(x, np.float32); return np.nan_to_num((x - x.mean(0)) / np.where(x.std(0) < 1e-6, 1, x.std(0)))


def colcorr(a, b):
    a = a - a.mean(0); b = b - b.mean(0); d = np.sqrt((a * a).sum(0) * (b * b).sum(0)); r = np.full(a.shape[1], np.nan)
    r[d > 1e-12] = (a * b).sum(0)[d > 1e-12] / d[d > 1e-12]; return np.clip(r, -.999999, .999999)


def ridge(xtr, ytr, xte, alpha):
    xm, ym = xtr.mean(0), ytr.mean(0); xc = xtr - xm
    return (xte - xm) @ np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ (ytr - ym)) + ym


def choose_alpha(x, y, stories, tol=1e-4):
    score = [np.mean([np.nanmean(np.arctanh(colcorr(ridge(np.concatenate([x[s] for s in stories if s != v]), np.concatenate([y[s] for s in stories if s != v]), x[v], a), y[v])))
                      for v in stories]) for a in ALPHAS]
    return next(a for a, s in zip(ALPHAS, score) if s >= np.nanmax(score) - tol)


def encode_participant(features, betas, target_features, target_groups, region_labels, seed=20260830):

    stories = sorted(features); regions = np.unique(region_labels[region_labels > 0]); out = {L: 0 for L in target_features}
    for held in stories:
        train = [s for s in stories if s != held]; pca = PCA(64, svd_solver="randomized", random_state=seed).fit(np.concatenate([features[s] for s in train]))
        tr = pca.transform(np.concatenate([features[s] for s in train])); mu, sd = tr.mean(0), np.where(tr.std(0) < 1e-6, 1, tr.std(0))
        xs = {s: (pca.transform(features[s]) - mu) / sd for s in stories}
        yreg = {s: np.stack([betas[s][:, region_labels == r].mean(1) for r in regions], 1) for s in stories}
        alpha = choose_alpha(xs, yreg, train)
        xtr, ytr = np.concatenate([xs[s] for s in train]), np.concatenate([betas[s] for s in train])
        for L in target_features:
            pred = ridge(xtr, ytr, (pca.transform(target_features[L][held]) - mu) / sd, alpha)
            out[L] = out[L] + np.nan_to_num(np.arctanh(colcorr(pred, target_groups[L][held])))
    return {L: (v / len(stories)).astype(np.float32) for L, v in out.items()}
