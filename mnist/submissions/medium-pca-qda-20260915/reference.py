#!/usr/bin/env python3
"""Ordered-FP32 PCA-QDA for MNIST-medium (5% error target).

Every operation is an IEEE FP32 add/sub/mul/div/compare/select in the same
order as the spatial-computer program built by spatial_program.py, so the two
are bitwise equal:

  features   u = sqrt(x) elementwise by NR_ITERATIONS Newton-Raphson steps from
             the seed x + 1e-6 (y <- 0.5 * (y + x / y)); x is the area-resized
             9x9 image / 255.
  basis      global mean m and packed second moments of the first NP training
             samples in supplied order; S = M/NP - m m^T (81x81);
             W0 = fixed literal 81xK matrix (numpy PCG64 seed 0 standard normals);
             ROUNDS rounds of subspace iteration: V = S W, then Gram-Schmidt
             without normalization (V_j -= (V_i.V_j)/(V_i.V_i) V_i for i < j,
             division only), then each column divided by its max |entry|; W = V.
             Column scaling is part of the frozen procedure: trace shrinkage
             makes the fitted classifier depend on the scale of the basis.
  projection z = (u - m) W  (K features per sample; sequential dot products)
  training   masked class counts, sums and packed second moments of z in one
             pass over the supplied order (mask = label == c via compare/select),
             mu = sum/count, S = M/count - mu mu^T, S <- (1-SHRINK) S + SHRINK (tr S / K) I,
             P = S^{-1} by Gauss-Jordan without pivoting,
             kappa = log(count/N) - 0.5 * sum log(pivots)
  prediction score_c = kappa_c - 0.5 * d^T P_c d with d = z_q - mu_c, using the packed
             upper triangle of P with doubled off-diagonals; first strict maximum wins.

log(x) is lowered to add/mul/div/compare/select: 40 fixed halving/doubling
steps bring x into [0.75, 1.5], then log(x) = k*ln2 + 2*atanh((x-1)/(x+1))
with the atanh series truncated after z^15.
"""
import numpy as np

f32 = np.float32
C, D, K = 10, 81, 40
NP = 2000                 # samples used for the global covariance (basis)
ROUNDS = 3                # subspace-iteration rounds
NR_ITERATIONS = 6         # Newton-Raphson sqrt steps
TINY = f32(1e-6)
SHRINK = f32(0.05)        # S <- (1-SHRINK) S + SHRINK * (tr S / K) I
LOG_STEPS = 40
SERIES_TERMS = 8          # z, z^3/3, ..., z^15/15
LN2 = f32(0.6931471805599453)
TRI81 = [(i, j) for i in range(D) for j in range(i, D)]    # 3321 packed entries
TRI81_I = np.array([i for i, j in TRI81]); TRI81_J = np.array([j for i, j in TRI81])
TRI = [(i, j) for i in range(K) for j in range(i, K)]      # 820 packed entries
TRI_I = np.array([i for i, j in TRI]); TRI_J = np.array([j for i, j in TRI])


def initial_basis():
    """The fixed literal start W0 (81 x K FP32); its raw bits are set-immediates in the grid program."""
    return np.random.default_rng(0).standard_normal((D, K)).astype(f32)


def nr_sqrt(x):
    """Elementwise Newton-Raphson sqrt: y = x + TINY; NR_ITERATIONS times q = x / y; q = y + q; y = 0.5 * q."""
    x = np.asarray(x, dtype=f32); y = x + TINY; half = f32(0.5)
    for _ in range(NR_ITERATIONS):
        q = x / y; q = y + q; y = half * q
    return y


def f32_log(x):
    """Elementwise FP32 log by range reduction and the atanh series (vectorized)."""
    x = np.asarray(x, dtype=f32); k = np.zeros_like(x)
    one, two, half, hi, lo = f32(1), f32(2), f32(0.5), f32(1.5), f32(0.75)
    for _ in range(LOG_STEPS):
        big = hi < x; x = np.where(big, x * half, x); k = np.where(big, k + one, k)
    for _ in range(LOG_STEPS):
        small = x < lo; x = np.where(small, x * two, x); k = np.where(small, k - one, k)
    z = (x - one) / (x + one); z2 = z * z; term = z; acc = z
    for m in range(1, SERIES_TERMS):
        term = term * z2; acc = acc + term / f32(2 * m + 1)
    return k * LN2 + two * acc


def gauss_jordan_inverse(S):
    """Inverse of an SPD KxK FP32 matrix without pivoting; returns (P, pivots)."""
    n = S.shape[0]; A = S.astype(f32).copy(); P = np.eye(n, dtype=f32); pivots = np.zeros(n, dtype=f32)
    for p in range(n):
        piv = A[p, p]; pivots[p] = piv
        for j in range(n):
            A[p, j] = A[p, j] / piv; P[p, j] = P[p, j] / piv
        for i in range(n):
            if i == p: continue
            fct = A[i, p]
            for j in range(n):
                A[i, j] = A[i, j] - fct * A[p, j]; P[i, j] = P[i, j] - fct * P[p, j]
    return P, pivots


def sdot(a, b):
    """Sequential FP32 dot product (index order), as the grid program accumulates it."""
    acc = f32(0)
    for i in range(len(a)):
        acc = acc + a[i] * b[i]
    return acc


def basis(u):
    """Global mean m and basis W from the first NP rows of u (N x 81 FP32)."""
    gsum = np.zeros(D, dtype=f32); gm = np.zeros(len(TRI81), dtype=f32)
    for n in range(NP):
        un = u[n]; gsum = gsum + un; gm = gm + (un[TRI81_I] * un[TRI81_J]).astype(f32)
    m = gsum / f32(NP)
    S = np.zeros((D, D), dtype=f32)
    for t, (i, j) in enumerate(TRI81):
        S[i, j] = gm[t] / f32(NP) - m[i] * m[j]; S[j, i] = S[i, j]
    W = initial_basis(); zero = f32(0)
    for _ in range(ROUNDS):
        V = np.zeros((D, K), dtype=f32)
        for i in range(D):
            for k in range(K):
                V[i, k] = sdot(S[i], W[:, k])
        den = np.zeros(K, dtype=f32)
        for j in range(K):
            for i in range(j):
                f = sdot(V[:, i], V[:, j]) / den[i]
                for r in range(D):
                    V[r, j] = V[r, j] - f * V[r, i]
            den[j] = sdot(V[:, j], V[:, j])
        for j in range(K):
            mx = zero
            for r in range(D):
                a = (zero - V[r, j]) if V[r, j] < zero else V[r, j]
                if mx < a: mx = a
            for r in range(D):
                V[r, j] = V[r, j] / mx
        W = V
    return m, W


def project(u, m, W):
    """z = (u - m) W with sequential FP32 dot products; u: (n, 81)."""
    d = (u - m).astype(f32); z = np.zeros((len(u), K), dtype=f32)
    for k in range(K):
        acc = np.zeros(len(u), dtype=f32)
        for i in range(D):
            acc = acc + d[:, i] * W[i, k]
        z[:, k] = acc
    return z


def train(x, labels):
    """x: (N, 81) FP32 pixels/255; labels: (N,) ints.  Returns the packed parameters."""
    N = len(x); u = nr_sqrt(x); m, W = basis(u); z = project(u, m, W)
    count = np.zeros(C, dtype=f32); ssum = np.zeros((C, K), dtype=f32); mom = np.zeros((C, len(TRI)), dtype=f32)
    one, zero = f32(1), f32(0)
    for n in range(N):                       # supplied sample order, sequential FP32 accumulation
        zn = z[n]; prod = (zn[TRI_I] * zn[TRI_J]).astype(f32)
        for c in range(C):
            msk = labels[n] == c
            count[c] = count[c] + (one if msk else zero)
            ssum[c] = ssum[c] + (zn if msk else zero)
            mom[c] = mom[c] + (prod if msk else zero)
    mu = np.zeros((C, K), dtype=f32); P = np.zeros((C, K, K), dtype=f32); kappa = np.zeros(C, dtype=f32)
    packed = np.zeros((C, len(TRI)), dtype=f32)
    for c in range(C):
        mu[c] = ssum[c] / count[c]
        S = np.zeros((K, K), dtype=f32)
        for t, (i, j) in enumerate(TRI):
            S[i, j] = mom[c, t] / count[c] - mu[c, i] * mu[c, j]; S[j, i] = S[i, j]
        tr = f32(0)
        for i in range(K): tr = tr + S[i, i]
        ridge = SHRINK * (tr / f32(K)); keep = f32(1) - SHRINK
        for t, (i, j) in enumerate(TRI):
            v = keep * S[i, j]
            if i == j: v = v + ridge
            S[i, j] = v; S[j, i] = v
        Pc, piv = gauss_jordan_inverse(S); P[c] = Pc
        logdet = f32(0)
        for lp in f32_log(piv): logdet = logdet + lp
        prior = f32_log(count[c] / f32(N))
        kappa[c] = prior - f32(0.5) * logdet
        for t, (i, j) in enumerate(TRI):
            packed[c, t] = Pc[i, j] if i == j else Pc[i, j] + Pc[i, j]
    return dict(m=m, W=W, mu=mu, packed=packed, kappa=kappa, count=count, P=P)


def predict(params, q):
    """q: (Q, 81) FP32 pixels/255.  Returns (scores (Q, C), labels (Q,))."""
    zq = project(nr_sqrt(q), params['m'], params['W']); Q = len(q)
    mu, packed, kappa = params['mu'], params['packed'], params['kappa']
    scores = np.zeros((Q, C), dtype=f32); half = f32(0.5)
    for c in range(C):
        d = zq - mu[c]
        s = np.zeros(Q, dtype=f32); t = 0
        for i in range(K):
            a = np.zeros(Q, dtype=f32)
            for j in range(i, K):
                a = a + packed[c, t] * d[:, j]; t += 1
            s = s + d[:, i] * a
        scores[:, c] = kappa[c] - half * s
    best = scores[:, 0].copy(); label = np.zeros(Q, dtype=np.int64)
    for c in range(1, C):
        better = best < scores[:, c]; best = np.where(better, scores[:, c], best); label = np.where(better, c, label)
    return scores, label


def train_predict(x, labels, q):
    params = train(x, labels); scores, pred = predict(params, q)
    return params, scores, pred


if __name__ == '__main__':
    raise SystemExit('use run.py (official draws) or pilot.py (pilot seeds)')
