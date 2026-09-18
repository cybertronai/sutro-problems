"""Ordered FP32 9-64-10 SGD learner for MNIST-small (self-contained).

Semantics match the reviewed SGD path of the shared grid scorer
(../grid-mlp-scoring-20260912/): squared-error loss, batch 25, constant
learning rate with step = lr/batch as one FP32 scalar, gradients from
pre-update weights, ascending-index FP32 reductions without fused
multiply-add, strict ReLU comparison, first-index argmax.
"""
import math
import numpy as np

f32 = np.float32
CONFIG = {'width': 64, 'epochs': 500, 'lr': 0.1, 'batch': 25}


def ordered_mm(a, b):
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        out = out + a[:, k, None] * b[None, k, :]
    return out


def ordered_rows(a):
    out = np.zeros(a.shape[1], dtype=np.float32)
    for row in a:
        out = out + row
    return out


def parameters(width, seed=101):
    rng = np.random.Generator(np.random.PCG64(seed))
    return [rng.uniform(-1 / 3, 1 / 3, (9, width)).astype(f32),
            np.zeros(width, dtype=f32),
            rng.uniform(-1 / math.sqrt(width), 1 / math.sqrt(width),
                        (width, 10)).astype(f32),
            np.zeros(10, dtype=f32)]


def transform(images):
    return images.reshape(len(images), 9) * f32(4) - f32(.5)


def train_predict(x, q, target, config, mm=ordered_mm, rows=ordered_rows):
    w, e, lr, b = config['width'], config['epochs'], config['lr'], config['batch']
    w1, b1, w2, b2 = parameters(w)
    step, zero = f32(lr / b), f32(0)
    for _ in range(e):
        for start in range(0, len(x), b):
            xb, tb = x[start:start + b], target[start:start + b]
            z = mm(xb, w1) + b1
            h = np.where(z > zero, z, zero)
            d2 = mm(h, w2) + b2 - tb
            d1 = np.where(z > zero, mm(d2, w2.T), zero)
            g1, gb1 = mm(xb.T, d1), rows(d1)
            g2, gb2 = mm(h.T, d2), rows(d2)
            w1 = w1 - step * g1
            b1 = b1 - step * gb1
            w2 = w2 - step * g2
            b2 = b2 - step * gb2
    zh = mm(q, w1) + b1
    scores = mm(np.where(zh > zero, zh, zero), w2) + b2
    return (np.concatenate([a.ravel() for a in (w1, b1, w2, b2)]).astype(f32),
            scores, scores.argmax(1).astype(np.int64))
