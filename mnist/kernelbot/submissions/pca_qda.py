"""PCA-QDA (medium-pca-qda-20260915 by Juraj Selep) ported to the KernelBot custom_kernel interface.

The learner functions below are copied unchanged from that submission's
gpu_benchmark.py (commit d1af086). Only the wrapper is new: the first call warms
up and captures one CUDA graph over the harness's fixed input tensors; later
calls replay it. Each replay fits the basis and QDA model and predicts all
queries, so nothing learned carries between calls.
"""
import numpy as np
import torch

C, D, K = 10, 81, 40
NP, ROUNDS, NR_ITERATIONS, TINY, SHRINK = 2000, 3, 6, 1e-6, 0.05

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def initial_basis():
    """The fixed literal start W0 from the submission's reference.py (NumPy PCG64 seed 0)."""
    return np.random.default_rng(0).standard_normal((D, K)).astype(np.float32)


def gauss_jordan(S):
    """Batched inverse of SPD (C,K,K) matrices without pivoting, as in reference.py; returns (P, log det)."""
    n = S.shape[1]; A = S.clone(); P = torch.eye(n, device=S.device, dtype=S.dtype).expand(S.shape[0], n, n).clone()
    logdet = torch.zeros(S.shape[0], device=S.device, dtype=S.dtype)
    for p in range(n):
        piv = A[:, p, p].clone(); logdet = logdet + torch.log(piv)
        A[:, p, :] = A[:, p, :] / piv[:, None]; P[:, p, :] = P[:, p, :] / piv[:, None]
        fct = A[:, :, p].clone(); fct[:, p] = 0
        A = A - fct[:, :, None] * A[:, p, :][:, None, :]; P = P - fct[:, :, None] * P[:, p, :][:, None, :]
    return P, logdet


def nr_sqrt(x):
    y = x + TINY
    for _ in range(NR_ITERATIONS): y = 0.5 * (y + x / y)
    return y


def basis_gpu(u, W0):
    """m and W by ROUNDS rounds of subspace iteration with column-wise Gram-Schmidt (classical form, vectorized
    over the previous columns) and max-|entry| scaling; same subspace construction as reference.py, not bitwise."""
    up = u[:NP]; m = up.mean(0); d = up - m; S = d.T @ d / NP; W = W0
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


def pca_qda_gpu(x, y, q, W0, scratch=None):
    """x (N,81) float32 cuda (pixels/255), y (N,) int64 cuda, q (Q,81).  Returns labels (Q,) int64."""
    N = x.shape[0]
    u = nr_sqrt(x); m, W = basis_gpu(u, W0); z = (u - m) @ W; zq = (nr_sqrt(q) - m) @ W
    onehot = torch.nn.functional.one_hot(y, C).to(torch.float32)          # (N, C)
    count = onehot.sum(0)                                                   # (C,)
    mu = (onehot.T @ z) / count[:, None]                                    # (C, K)
    zz = z[:, :, None] * z[:, None, :]                                      # (N, K, K)
    mom = torch.einsum('nc,nij->cij', onehot, zz) / count[:, None, None]    # (C, K, K)
    S = mom - mu[:, :, None] * mu[:, None, :]
    tr = torch.diagonal(S, dim1=1, dim2=2).sum(1)
    S = (1 - SHRINK) * S + (SHRINK * tr / K)[:, None, None] * torch.eye(K, device=S.device)
    P, logdet = gauss_jordan(S)                                             # capturable, no cuSOLVER
    kappa = torch.log(count / N) - 0.5 * logdet                             # (C,)
    d = zq[:, None, :] - mu[None, :, :]                                     # (Q, C, K)
    s = torch.einsum('qci,cij,qcj->qc', d, P, d)
    scores = kappa[None, :] - 0.5 * s
    if scratch is not None:
        # Retain already-live tensors for poisoning outside graph capture.
        # This adds no GPU operations to the captured learning task.
        scratch.update(u=u, m=m, W=W, z=z, zq=zq, onehot=onehot, count=count,
                       mu=mu, zz=zz, mom=mom, S=S, P=P, logdet=logdet,
                       kappa=kappa, d=d, s=s, scores=scores)
    return scores.argmax(1)


_graph = None


def _capture(train_x, train_y, test_x):
    global _graph
    w0 = torch.tensor(initial_basis(), device='cuda')
    x, q = train_x.view(train_x.shape[0], D), test_x.view(test_x.shape[0], D)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pca_qda_gpu(x, train_y, q, w0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = pca_qda_gpu(x, train_y, q, w0)
    _graph = (graph, output, train_x.data_ptr(), train_y.data_ptr(), test_x.data_ptr())


def custom_kernel(data):
    train_x, train_y, test_x = data
    pointers = (train_x.data_ptr(), train_y.data_ptr(), test_x.data_ptr())
    if _graph is None or _graph[2:] != pointers:
        _capture(train_x, train_y, test_x)
    _graph[0].replay()
    return _graph[1]
