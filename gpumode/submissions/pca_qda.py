#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

"""PCA-QDA, ported from mnist/submissions/medium-pca-qda-20260915 (@jurajselep).

The learner is unchanged from that submission's gpu_benchmark.py: an arcsine
pixel transform by Newton square roots, a 40-dimensional subspace found by three
rounds of subspace iteration from a fixed seeded Gaussian start, then quadratic
discriminant analysis with a shrunk covariance and a Gauss-Jordan inverse (no
cuSOLVER, so the whole thing is CUDA-graph capturable). About 95% accurate on
MNIST-medium; it was the 5% band leader at roughly 3.3 ms and 174 mJ per call.

Only the wrapper is new. On CUDA the first call captures one graph over the
harness's fixed input tensors and later calls replay it; each replay refits the
basis and the QDA model from the training rows that are in the tensors at that
moment, so nothing learned survives between calls. On CPU it simply runs.

The starting basis W0 is numpy.random.default_rng(0).standard_normal((D, 40)):
a seeded random initialization, not a trained constant.

Harness 1.2.0 release. Draws now arrive as (N, D) features -- the draw's images
whitened onto D = 60 principal directions and secretly rotated -- rather than
(N, 1, 9, 9) pixels. QDA itself is unchanged; the two pixel-specific steps in
front of it are switched off, because on that input they do nothing:

  * the arcsine transform (the Newton square root) is a variance-stabilizing
    transform for counts in [0, 1]. The release is signed, where the Newton
    iteration does not converge, so it is applied only to a pixel release.
  * the subspace iteration finds the top-variance directions. The release is
    already the top-60 principal subspace, and it is white, so every direction
    has the same variance and there is nothing left to rank: the iteration would
    return an arbitrary 40-dimensional subspace and throw a third of the signal
    away (measured: 90.2% against 90.7% for keeping all 60 on a 10,000-example
    draw). On a linear release the model therefore runs on all D coordinates.

Measured on one 10,000/10,000 CPU draw: 95.7% on the pixel release, 90.7% on the
60-dimensional linear release. Exact whitening flattens the variance spectrum
that QDA's isotropic shrinkage was implicitly using to damp the noisy
directions, and that is what the 5 points buy.
"""

import numpy as np
import torch

C, K = 10, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def initial_basis(features):
    """The fixed seeded start W0 from the submission's reference.py (NumPy PCG64 seed 0)."""
    return np.random.default_rng(0).standard_normal((features, K)).astype(np.float32)


def gauss_jordan(S):
    """Batched inverse of SPD (C,K,K) matrices without pivoting; returns (inverse, log det)."""
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


def keep_all(u, W0):
    """The basis step for an already-whitened release: centre, keep every column."""
    return u[:NP].mean(0), W0


def basis_gpu(u, W0):
    """Subspace iteration with column-wise Gram-Schmidt and max-|entry| scaling."""
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


def pca_qda(x, y, q, W0, transform, basis):
    """x (N,D) float32, y (N,) int64, q (Q,D) -> labels (Q,) int64."""
    N = x.shape[0]
    u = transform(x)
    m, W = basis(u, W0)
    k = W.shape[1]
    z = (u - m) @ W
    zq = (transform(q) - m) @ W
    onehot = torch.nn.functional.one_hot(y, C).to(torch.float32)
    count = onehot.sum(0)
    mu = (onehot.T @ z) / count[:, None]
    zz = z[:, :, None] * z[:, None, :]
    mom = torch.einsum("nc,nij->cij", onehot, zz) / count[:, None, None]
    S = mom - mu[:, :, None] * mu[:, None, :]
    tr = torch.diagonal(S, dim1=1, dim2=2).sum(1)
    S = (1 - SHRINK) * S + (SHRINK * tr / k)[:, None, None] * torch.eye(k, device=S.device)
    P, logdet = gauss_jordan(S)
    kappa = torch.log(count / N) - 0.5 * logdet
    d = zq[:, None, :] - mu[None, :, :]
    s = torch.einsum("qci,cij,qcj->qc", d, P, d)
    return (kappa[None, :] - 0.5 * s).argmax(1)


_graph = None
_w0 = {}


def identity(values):
    return values


def plan(train_x):
    """(W0, input transform, basis step) for the release this draw arrived in.

    A pixel release is (N, 1, size, size) in [0, 1]: the upstream arcsine
    transform and the seeded 40-dimensional subspace iteration. A linear release
    is (N, D), already centred, whitened and rotated: no transform and no
    projection.
    """
    device = train_x.device
    features = train_x.reshape(train_x.shape[0], -1).shape[1]
    spatial = train_x.dim() == 4
    key = (str(device), features, spatial)
    if key not in _w0:
        _w0[key] = (
            torch.as_tensor(initial_basis(features), device=device)
            if spatial
            else torch.eye(features, device=device)
        )
    if spatial:
        return _w0[key], nr_sqrt, basis_gpu
    return _w0[key], identity, keep_all


def _capture(train_x, train_y, test_x, w0, transform, basis):
    global _graph
    x = train_x.view(train_x.shape[0], -1)
    q = test_x.view(test_x.shape[0], -1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pca_qda(x, train_y, q, w0, transform, basis)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = pca_qda(x, train_y, q, w0, transform, basis)
    _graph = (graph, output, train_x.data_ptr(), train_y.data_ptr(), test_x.data_ptr())


def custom_kernel(data):
    global _graph
    train_x, train_y, test_x = data
    w0, transform, basis = plan(train_x)
    if train_x.device.type != "cuda":
        return pca_qda(
            train_x.reshape(train_x.shape[0], -1),
            train_y,
            test_x.reshape(test_x.shape[0], -1),
            w0,
            transform,
            basis,
        )
    pointers = (train_x.data_ptr(), train_y.data_ptr(), test_x.data_ptr())
    if _graph is None or _graph[2:] != pointers:
        _capture(train_x, train_y, test_x, w0, transform, basis)
    _graph[0].replay()
    return _graph[1]
