"""PCANet nine-block features, PCA-100, and eight-component class mixtures.

This extracts the selected learner from bench_two.py, fast_pcanet.py and
fast4.py without their measurement, data-loading or process-wide side effects.
All inputs are already-normalized FP32 tensors; no test labels are accepted.
The caller controls TF32/cuDNN settings and records them in the run evidence.

Both train_predict backends refit every learned parameter. They share the
optimized FP32 PCA projection and fast mixture head. The reference backend
uses the original PyTorch feature extractor; the CUDA backend replaces only
stage-two convolution and histogram construction with the fused CUDA kernel.
The two extractors can differ around zero because of convolution arithmetic.

The original slower mixture is exposed as mixture for separate audit. It is
not the head used by either train_predict backend. Learned filter fitting
retains FP64 covariance/eigendecomposition and the original sample prefixes.
"""

import math

import torch
import torch.nn.functional as F

CLS = 10
L1 = 8
L2 = 5
MIXTURE_COMPONENTS = 8



def pca_filters(P, L):
    X = P.reshape(-1, P.shape[-1]).double()
    if len(X) > 200000: X = X[:200000]
    ev, V = torch.linalg.eigh(X.T @ X / len(X))
    return V[:, -L:].float()


def patches(img, k):
    p = k // 2
    u = F.unfold(F.pad(img.unsqueeze(1), (p, p, p, p)), k).transpose(1, 2)
    return u - u.mean(-1, keepdim=True)


def stage(img, W, k):
    """Center each filter to apply centered-patch convolution without unfolding."""
    Wc = (W - W.mean(0, keepdim=True)).T.reshape(-1, 1, k, k).contiguous()
    return F.conv2d(img.unsqueeze(1) if img.dim() == 3 else img, Wc, padding=k // 2)


def pcanet_feats(x, W1, W2, L1, L2, blk=14, stride=7, chunk=5000):
    outs = []
    nb = 1 << L2
    org = [(r, c) for r in range(0, 28 - blk + 1, stride) for c in range(0, 28 - blk + 1, stride)]
    for s in range(0, len(x), chunk):
        img = x[s:s + chunk]; n = img.shape[0]
        M1 = stage(img, W1, 7)
        code = torch.zeros(n, L1, 28, 28, device=x.device)
        for i in range(L1):
            S = stage(M1[:, i:i + 1], W2, 7)
            for b in range(L2): code[:, i] += (1 << b) * (S[:, b] > 0).float()
        hist = torch.zeros(n, L1, len(org), nb, device=x.device)
        ci = code.long()
        for j, (r0, c0) in enumerate(org):
            v = ci[:, :, r0:r0 + blk, c0:c0 + blk].reshape(n, L1, -1)
            hist[:, :, j] = torch.zeros(n, L1, nb, device=x.device).scatter_add_(
                2, v, torch.ones_like(v, dtype=torch.float32))
        h = hist.reshape(n, -1)
        outs.append(torch.sqrt(h / (h.sum(1, keepdim=True) + 1e-9)))
    return torch.cat(outs, 0)


def learn_filters(x):
    """Refit two 7x7 filter banks from the original fixed training prefixes.

    Bank one uses the first 255 images; pca_filters caps its covariance at
    200,000 centered patches. Bank two starts with all eight stage-one planes
    from the first 250 images, keeps the first 2,000 planes, and applies the
    same 200,000-patch cap. No fitted filter is cached across calls.
    """
    W1 = pca_filters(patches(x[:255], 7), L1)
    M1 = stage(x[:250], W1, 7)
    W2 = pca_filters(patches(M1.reshape(-1, 28, 28)[:2000], 7), L2)
    return W1, W2


def feats_cuda(x, W1, W2, L1, L2, chunk=15000):
    # Keep extension loading lazy: importing this module does not compile CUDA.
    if __package__:
        from .cuda_kernel import cuda_hist
    else:
        from cuda_kernel import cuda_hist
    W2c = (W2 - W2.mean(0, keepdim=True)).T.reshape(L2, 1, 7, 7).contiguous()
    W2g = W2c.repeat(L1, 1, 1, 1)
    outs = []
    for s in range(0, len(x), chunk):
        M1 = stage(x[s:s + chunk], W1, 7).contiguous()
        h = cuda_hist(M1, W2g, L1, L2).reshape(M1.shape[0], -1)   # conv, threshold, pack, histogram
        outs.append(torch.sqrt(h / (h.sum(1, keepdim=True) + 1e-9)))
        del M1
    return torch.cat(outs, 0)


def pca_project(Ftr, Fte, K):
    """Original optimized head: centered FP32 covariance and FP32 eigensolve."""
    m = Ftr.mean(0)
    dd = Ftr - m
    _, V = torch.linalg.eigh((dd.T @ dd) / len(Ftr))
    W = V[:, -K:]
    return dd @ W, (Fte - m) @ W


def mixture(z, y, zq, k, NC=CLS, lam=0.5, steps=8, shrink=0.05, ridge=1e-4):
    K = z.shape[1]; Id = torch.eye(K, device=z.device); out = torch.zeros(len(zq), NC, device=z.device)
    for c in range(NC):
        zc = z[y == c]; n = len(zc); mu_c = zc.mean(0)
        cov_c = ((zc - mu_c).T @ (zc - mu_c)) / n
        cov_c = (1 - shrink) * cov_c + shrink * torch.trace(cov_c) / K * Id
        g = torch.Generator(device='cpu').manual_seed(900 + c)
        ctr = zc[torch.randperm(n, generator=g)[:k].to(z.device)].clone()
        for _ in range(8):
            a = torch.cdist(zc, ctr).argmin(1)
            for j in range(k):
                m = a == j
                if m.sum() > 1: ctr[j] = zc[m].mean(0)
        Rw = F.one_hot(a, k).float()
        for step in range(steps + 1):
            Nj = Rw.sum(0) + 1e-9; pi = Nj / n
            mu = (Rw.T @ zc) / Nj[:, None]
            dif = zc[:, None, :] - mu[None]
            Sj = torch.einsum('nki,nkj,nk->kij', dif, dif, Rw) / Nj[:, None, None]
            Sj = (1 - lam) * Sj + lam * cov_c[None] + ridge * Id[None]
            L = torch.linalg.cholesky(Sj)
            ld = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
            if step == steps: break
            sol = torch.cholesky_solve(dif.permute(1, 2, 0), L)
            qd = torch.einsum('nki,kin->nk', dif, sol)
            ll = torch.log(pi)[None] - 0.5 * qd - 0.5 * ld[None]
            Rw = torch.softmax(ll, 1)
        sc = torch.empty(len(zq), device=z.device)
        for i in range(0, len(zq), 2500):
            d = zq[i:i + 2500, None, :] - mu[None]
            sol = torch.cholesky_solve(d.permute(1, 2, 0), L)
            qd = torch.einsum('nki,kin->nk', d, sol)
            llq = torch.log(pi)[None] - 0.5 * qd - 0.5 * ld[None]
            sc[i:i + 2500] = torch.logsumexp(llq, 1)
        out[:, c] = sc + math.log(n / len(z))
    return out


def fast_mixture(z, y, zq, k, NC=CLS, lam=0.5, steps=8, shrink=0.05, ridge=1e-4):
    """Optimized mixture: batched products and triangular solves (FP32).

    This preserves the source implementation, including eight k-means passes,
    per-class CPU seeds 900+c, eight EM updates followed by a final refit,
    covariance shrinkage and ridge. Reduction order differs from mixture().
    """
    K = z.shape[1]; Id = torch.eye(K, device=z.device); out = torch.empty(len(zq), NC, device=z.device)
    for c in range(NC):
        zc = z[y == c]; n = len(zc); mu_c = zc.mean(0)
        cov_c = ((zc - mu_c).T @ (zc - mu_c)) / n
        cov_c = (1 - shrink) * cov_c + shrink * torch.trace(cov_c) / K * Id
        g = torch.Generator(device='cpu').manual_seed(900 + c)
        ctr = zc[torch.randperm(n, generator=g)[:k].to(z.device)].clone()
        for _ in range(8):
            a = torch.cdist(zc, ctr).argmin(1)
            oh = F.one_hot(a, k).float()
            cnt = oh.sum(0).clamp(min=1)
            ctr = torch.where((oh.sum(0) > 1)[:, None], (oh.T @ zc) / cnt[:, None], ctr)
        Rw = F.one_hot(a, k).float()
        for step in range(steps + 1):
            Nj = Rw.sum(0) + 1e-9; pi = Nj / n
            mu = (Rw.T @ zc) / Nj[:, None]
            d = zc[:, None, :] - mu[None]                       # (n, k, K)
            dt = d.permute(1, 2, 0)                             # (k, K, n)
            Sj = torch.bmm(dt * Rw.T[:, None, :], dt.transpose(1, 2)) / Nj[:, None, None]
            Sj = (1 - lam) * Sj + lam * cov_c[None] + ridge * Id[None]
            L = torch.linalg.cholesky(Sj)
            ld = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
            if step == steps: break
            w = torch.linalg.solve_triangular(L, dt, upper=False)   # (k, K, n)
            qd = (w * w).sum(1).T                                   # (n, k)
            Rw = torch.softmax(torch.log(pi)[None] - 0.5 * qd - 0.5 * ld[None], 1)
        dq = (zq[:, None, :] - mu[None]).permute(1, 2, 0)
        w = torch.linalg.solve_triangular(L, dq, upper=False)
        qd = (w * w).sum(1).T
        out[:, c] = torch.logsumexp(torch.log(pi)[None] - 0.5 * qd - 0.5 * ld[None], 1) + math.log(n / len(z))
    return out


@torch.no_grad()
def train_predict(x, y, q, K=100, backend="cuda"):
    """Train from x/y and predict q; return one integer label per query.

    x and q have shape (N, 28, 28), dtype float32, and values normalized to
    [0, 1]. y is an int64 label vector. All tensors must share a device.
    backend="cuda" requires CUDA; backend="reference" also works on CPU.
    Input transfers, normalization and extension compilation belong to the
    calling harness. No learned state or labels persist between calls.
    """
    if backend not in ("cuda", "reference"):
        raise ValueError("backend must be 'cuda' or 'reference'")
    if x.ndim != 3 or q.ndim != 3 or tuple(x.shape[1:]) != (28, 28) or tuple(q.shape[1:]) != (28, 28):
        raise ValueError("x and q must have shape (N, 28, 28)")
    if x.dtype != torch.float32 or q.dtype != torch.float32:
        raise TypeError("x and q must be normalized float32 tensors")
    if y.ndim != 1 or len(y) != len(x) or y.dtype != torch.int64:
        raise ValueError("y must be an int64 vector with one label per training image")
    if x.device != y.device or x.device != q.device:
        raise ValueError("x, y and q must be on the same device")
    if backend == "cuda" and not x.is_cuda:
        raise ValueError("backend='cuda' requires CUDA input tensors")
    if not len(x) or not len(q):
        raise ValueError("training and query tensors must be nonempty")
    if not isinstance(K, int) or not 1 <= K <= L1 * 9 * (1 << L2):
        raise ValueError("K must be a positive integer no larger than 2304")
    W1, W2 = learn_filters(x)
    features = feats_cuda if backend == "cuda" else pcanet_feats
    Ftr = features(x, W1, W2, L1, L2)
    Fte = features(q, W1, W2, L1, L2)
    z, zq = pca_project(Ftr, Fte, K)
    return fast_mixture(z, y, zq, MIXTURE_COMPONENTS).argmax(1)
