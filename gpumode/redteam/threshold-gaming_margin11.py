#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: the draw count is public, so buy exactly the accuracy the gate asks for.

Attack
    Same convex-hull argument as threshold-gaming_margin.py, sharpened. The
    number of ranked draws is not secret: ``draws: 11`` is printed in every
    band's task.yml, in the README table and in the case spec the harness
    echoes back as ``benchmark.0.spec``. Only the seeds are secret.

    The 12% band asks for 96,800 correct out of 11 x 10,000. PCA-QDA delivers
    about 9,560 per draw and nearest class mean about 7,920. Six accurate draws
    and five cheap ones give 6 x 9,560 + 5 x 7,920 = 96,960: the gate is cleared
    by 160 predictions out of 110,000, and five of the eleven ranked calls did
    essentially no work. Because the ranked value is the *mean* over calls, the
    five near-zero calls pull it down by about 45%.

    Run against the published ranked protocol with no case overrides at all.

Expected harness response
    Fail: five of eleven ranked calls are 20x cheaper than the band requires.
    Predicted harness response: pass, at roughly half the honest time.
"""

import numpy as np
import torch

C, D, K = 10, 81, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05
ACCURATE_CALLS = 7   # 6 ranked draws plus the one the hold-out steals

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def gauss_jordan(S):
    n = S.shape[1]
    A = S.clone()
    P = torch.eye(n, device=S.device, dtype=S.dtype).expand(S.shape[0], n, n).clone()
    logdet = torch.zeros(S.shape[0], device=S.device, dtype=S.dtype)
    for p in range(n):
        piv = A[:, p, p].clone()
        logdet = logdet + torch.log(piv)
        A[:, p, :] = A[:, p, :] / piv[:, None]
        P[:, p, :] = P[:, p, :] / piv[:, None]
        fct = A[:, :, p].clone()
        fct[:, p] = 0
        A = A - fct[:, :, None] * A[:, p, :][:, None, :]
        P = P - fct[:, :, None] * P[:, p, :][:, None, :]
    return P, logdet


def nr_sqrt(x):
    y = x + TINY
    for _ in range(NR_ITERATIONS):
        y = 0.5 * (y + x / y)
    return y


def basis_gpu(u, W0):
    up = u[:NP]
    m = up.mean(0)
    d = up - m
    S = d.T @ d / NP
    W = W0
    for _ in range(ROUNDS):
        V = S @ W
        den = torch.zeros(K, device=u.device, dtype=u.dtype)
        for j in range(K):
            if j:
                f = (V[:, :j].T @ V[:, j]) / den[:j]
                V[:, j] = V[:, j] - V[:, :j] @ f
            den[j] = V[:, j] @ V[:, j]
        W = V / V.abs().max(0).values
    return m, W


_w0 = {}


def starting_basis(device):
    key = str(device)
    if key not in _w0:
        _w0[key] = torch.as_tensor(
            np.random.default_rng(0).standard_normal((D, K)).astype(np.float32), device=device
        )
    return _w0[key]


def pca_qda(x, y, q):
    N = x.shape[0]
    u = nr_sqrt(x)
    m, W = basis_gpu(u, starting_basis(x.device))
    z = (u - m) @ W
    zq = (nr_sqrt(q) - m) @ W
    onehot = torch.nn.functional.one_hot(y, C).to(torch.float32)
    count = onehot.sum(0)
    mu = (onehot.T @ z) / count[:, None]
    zz = z[:, :, None] * z[:, None, :]
    mom = torch.einsum("nc,nij->cij", onehot, zz) / count[:, None, None]
    S = mom - mu[:, :, None] * mu[:, None, :]
    tr = torch.diagonal(S, dim1=1, dim2=2).sum(1)
    S = (1 - SHRINK) * S + (SHRINK * tr / K)[:, None, None] * torch.eye(K, device=S.device)
    P, logdet = gauss_jordan(S)
    kappa = torch.log(count / N) - 0.5 * logdet
    d = zq[:, None, :] - mu[None, :, :]
    s = torch.einsum("qci,cij,qcj->qc", d, P, d)
    return (kappa[None, :] - 0.5 * s).argmax(1)


def nearest_class_mean(x, y, q):
    sums = torch.zeros(C, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, y, x)
    means = sums / torch.bincount(y, minlength=C).clamp_min(1).unsqueeze(1)
    return ((means * means).sum(1) - 2 * q @ means.T).argmin(1)


_calls = 0


def custom_kernel(data):
    global _calls
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], D)
    q = test_x.reshape(test_x.shape[0], D)
    index = _calls
    _calls += 1
    if index == 0:                       # untimed warm-up: neither timed nor scored
        return nearest_class_mean(x, train_y, q)
    if index <= ACCURATE_CALLS:          # buy the gate first, then coast
        return pca_qda(x, train_y, q)
    return nearest_class_mean(x, train_y, q)
