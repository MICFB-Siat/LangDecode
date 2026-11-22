import numpy as np, torch
from langdecoder.model import Net2, loss_fn

torch.manual_seed(0); rng = np.random.default_rng(0)
N_SUB, K_VOX, N_DELAY, DIM, N_TRAIN, N_TEST = 3, 500, 7, 1536, 1200, 120

z = rng.standard_normal((N_TRAIN + N_TEST, 32)).astype(np.float32)
Y = z @ rng.standard_normal((32, DIM)).astype(np.float32)
maps = rng.standard_normal((N_SUB, 32, K_VOX)).astype(np.float32)
sub = rng.integers(0, N_SUB, len(z))
X = np.stack([z[i] @ maps[sub[i]] for i in range(len(z))])[:, None].repeat(N_DELAY, 1)
X = X + 12.0 * rng.standard_normal(X.shape).astype(np.float32)
X, Y = torch.from_numpy(X), torch.from_numpy(Y)

net = Net2(N_SUB, K_VOX, N_DELAY, d=64, depth=2, n_head=4, n_layer=1, out=DIM, drop=0.1); net.reset_head()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
for step in range(300):
    k = step % N_SUB; idx = np.flatnonzero(sub[:N_TRAIN] == k); i = torch.from_numpy(rng.choice(idx, 128, replace=False))
    loss = loss_fn(net(X[i], k), Y[i], net.logit_scale)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == 299: print(f"step {step:3d}  loss {loss.item():.3f}")

net.eval()
with torch.no_grad():
    t = np.arange(N_TRAIN, len(z)); P = torch.zeros(N_TEST, DIM)
    for k in range(N_SUB): m = sub[t] == k; P[torch.from_numpy(m)] = net(X[t[m]], k)
    P = torch.nn.functional.normalize(P, dim=-1); T = torch.nn.functional.normalize(Y[N_TRAIN:], dim=-1)
C = (P @ T.T).numpy(); rank = np.mean([(C[k] < C[k, k]).sum() / (len(C) - 1) for k in range(len(C))])
print(f"held-out cosine {np.mean(np.diag(C)):.3f}; identification (normalized rank) {rank:.3f}, chance 0.5")
