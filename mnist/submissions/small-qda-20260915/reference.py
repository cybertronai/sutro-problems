#!/usr/bin/env python3
"""Ordered-FP32 quadratic discriminant analysis (QDA) for MNIST-small.

Every operation is an IEEE FP32 add/sub/mul/div in the same order as the
unoptimized spatial-computer lowering in spatial_program.py. The submitted
lowering removes normalization and expands the quadratic; its output labels
are checked separately against this reference on all 11 recorded draws:

  training   masked class counts, sums and second moments (one pass over the
             supplied sample order; mask = label == c via compare/select),
             mu = sum/count, S = M/count - mu mu^T (biased covariance),
             P = S^{-1} by Gauss-Jordan without pivoting (S is SPD),
             kappa = log(count/N) - 0.5 * sum_i log(pivot_i)   (log det S)
  prediction score_c = kappa_c - 0.5 * d^T P_c d with d = q - mu_c, using the
             packed upper triangle of P with doubled off-diagonals;
             label = first class attaining the maximum (strict less-than).

log(x) is lowered to add/mul/div/compare/select: 40 fixed halving/doubling
steps bring x into [0.75, 1.5], then log(x) = k*ln2 + 2*atanh((x-1)/(x+1))
with the atanh series truncated after z^15.
"""
import numpy as np

f32 = np.float32
C, D = 10, 9
LOG_STEPS = 40
SERIES_TERMS = 8          # z, z^3/3, ..., z^15/15
LN2 = f32(0.6931471805599453)
TRI = [(i, j) for i in range(D) for j in range(i, D)]   # 45 packed upper-triangle entries


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
    """Inverse of an SPD DxD FP32 matrix without pivoting; returns (P, pivots)."""
    A = S.astype(f32).copy(); P = np.eye(D, dtype=f32); pivots = np.zeros(D, dtype=f32)
    for p in range(D):
        piv = A[p, p]; pivots[p] = piv
        for j in range(D):
            A[p, j] = A[p, j] / piv; P[p, j] = P[p, j] / piv
        for i in range(D):
            if i == p: continue
            fct = A[i, p]
            for j in range(D):
                A[i, j] = A[i, j] - fct * A[p, j]; P[i, j] = P[i, j] - fct * P[p, j]
    return P, pivots


def train(x, labels):
    """x: (N, D) normalized FP32 pixels; labels: (N,) ints.  Returns packed parameters."""
    N = len(x); x = x.astype(f32)
    count = np.zeros(C, dtype=f32); ssum = np.zeros((C, D), dtype=f32); mom = np.zeros((C, len(TRI)), dtype=f32)
    one, zero = f32(1), f32(0)
    for n in range(N):                       # supplied sample order, sequential FP32 accumulation
        xi = x[n]; prod = np.array([xi[i] * xi[j] for i, j in TRI], dtype=f32)
        for c in range(C):
            m = labels[n] == c
            count[c] = count[c] + (one if m else zero)
            ssum[c] = ssum[c] + (xi if m else zero)
            mom[c] = mom[c] + (prod if m else zero)
    mu = np.zeros((C, D), dtype=f32); P = np.zeros((C, D, D), dtype=f32); kappa = np.zeros(C, dtype=f32)
    packed = np.zeros((C, len(TRI)), dtype=f32)
    for c in range(C):
        mu[c] = ssum[c] / count[c]
        S = np.zeros((D, D), dtype=f32)
        for t, (i, j) in enumerate(TRI):
            S[i, j] = mom[c, t] / count[c] - mu[c, i] * mu[c, j]; S[j, i] = S[i, j]
        Pc, piv = gauss_jordan_inverse(S); P[c] = Pc
        logdet = f32(0)
        for lp in f32_log(piv): logdet = logdet + lp
        prior = f32_log(count[c] / f32(N))
        kappa[c] = prior - f32(0.5) * logdet
        for t, (i, j) in enumerate(TRI):
            packed[c, t] = Pc[i, j] if i == j else Pc[i, j] + Pc[i, j]
    return dict(mu=mu, packed=packed, kappa=kappa, count=count, P=P)


def predict(params, q):
    """q: (Q, D) normalized FP32 queries.  Returns (scores (Q, C), labels (Q,))."""
    q = q.astype(f32); Q = len(q); mu, packed, kappa = params['mu'], params['packed'], params['kappa']
    scores = np.zeros((Q, C), dtype=f32); half = f32(0.5)
    for c in range(C):
        d = q - mu[c]                                    # 9 subs per query
        s = np.zeros(Q, dtype=f32); t = 0
        for i in range(D):
            a = np.zeros(Q, dtype=f32)
            for j in range(i, D):
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

