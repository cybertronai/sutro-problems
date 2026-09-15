"""Ordered FP32 H32 MLP math, adapted from accuracy-il-20260911."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np

BATCH = 25


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def ordered_mm(a, b):
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        out = out + a[:, k, None] * b[None, k, :]
    return out


def fast_mm(a, b):
    # A strided/transposed RHS can select a different einsum reduction kernel.
    # Materializing its row-major layout keeps the ascending-K operation order.
    return np.einsum('ik,kj->ij', a, np.ascontiguousarray(b),
                     optimize=False, dtype=np.float32)


def ordered_rows(a):
    out = np.zeros(a.shape[1], dtype=np.float32)
    for row in a:
        out = out + row
    return out


def fast_rows(a):
    return np.einsum('ij->j', a, optimize=False, dtype=np.float32)


def parameters(width, seed):
    # Random constants depend only on predeclared architecture and seed, never
    # on training or test examples. Their raw FP32 bits can be emitted as set.
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.uniform(-1 / 3, 1 / 3, (9, width)).astype(np.float32),
            np.zeros(width, dtype=np.float32),
            rng.uniform(-1 / math.sqrt(width), 1 / math.sqrt(width),
                        (width, 10)).astype(np.float32),
            np.zeros(10, dtype=np.float32)]


def transform(images):
    return images.reshape(len(images), 9) * np.float32(4) - np.float32(0.5)


def forward(x, params, mm=fast_mm):
    w1, b1, w2, b2 = params
    z = mm(x, w1) + b1
    h = np.where(z > np.float32(0), z, np.float32(0))
    return z, h, mm(h, w2) + b2


def predict(x, params, mm=fast_mm):
    # First-column tie breaking is the same as an ascending strict-> scan.
    return np.argmax(forward(x, params, mm)[2], axis=1).astype(np.int64)


def update(x, target, params, rate, mm=fast_mm, rows=fast_rows):
    w1, b1, w2, b2 = params
    z, h, scores = forward(x, params, mm)
    d2 = scores - target
    back = mm(d2, w2.T)
    d1 = np.where(z > np.float32(0), back, np.float32(0))
    g1, gb1 = mm(x.T, d1), rows(d1)
    g2, gb2 = mm(h.T, d2), rows(d2)
    # All four gradients use the same pre-update weights. Squared-error loss
    # is 0.5*sum_class(error**2), averaged across this complete minibatch.
    step = np.float32(rate / len(x))
    return [w1 - step * g1, b1 - step * gb1,
            w2 - step * g2, b2 - step * gb2]


def epoch(x, target, params, rate, mm=fast_mm, rows=fast_rows):
    for start in range(0, len(x), BATCH):
        params = update(x[start:start+BATCH], target[start:start+BATCH],
                        params, rate, mm, rows)
    return params


