"""Ordered FP32 NR-K8 Adam learner for MNIST-small (self-contained)."""
import math
from pathlib import Path
import numpy as np

f32 = np.float32
CONFIG = {'width': 32, 'epochs': 100, 'nr': 8, 'batch': 25, 'lr': 0.2}


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
    return [rng.uniform(-1 / 3, 1 / 3, (9, width)).astype(f32), np.zeros(width, f32),
            rng.uniform(-1 / np.sqrt(width), 1 / np.sqrt(width), (width, 10)).astype(f32), np.zeros(10, f32)]


def transform(images):
    return images.reshape(len(images), 9) * f32(4) - f32(.5)


def sqrt_nr(x, iterations):
    y = x + f32(1e-6)
    for _ in range(iterations):
        y = f32(.5) * (y + x / y)
    return y


def train_predict(x, q, target, config, mm=ordered_mm, rows=ordered_rows):
    w, e, b, n = config['width'], config['epochs'], config['batch'], config['nr']
    w1, b1, w2, b2 = parameters(w)
    step, eps = f32(config['lr'] / b), f32(1e-8)
    m = {k: np.zeros_like(v) for k, v in (('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2))}
    v = {k: np.zeros_like(x) for k, x in m.items()}
    t = 0
    for _ in range(e):
        for start in range(0, len(x), b):
            t += 1
            xb, tb = x[start:start+b], target[start:start+b]
            z = mm(xb, w1) + b1
            h = np.where(z > f32(0), z, f32(0))
            d2 = mm(h, w2) + b2 - tb
            d1 = np.where(z > f32(0), mm(d2, w2.T), f32(0))
            grads = {'w1': mm(xb.T, d1), 'b1': rows(d1), 'w2': mm(h.T, d2), 'b2': rows(d2)}
            params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
            for key in grads:
                g = grads[key]
                m[key] = f32(.9) * m[key] + f32(.1) * g
                v[key] = f32(.999) * v[key] + f32(.001) * g * g
                mhat = m[key] / (1 - f32(.9) ** t)
                vhat = v[key] / (1 - f32(.999) ** t)
                params[key] = params[key] - step * mhat / (sqrt_nr(vhat, n) + eps)
            w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']
    zh = mm(q, w1) + b1
    scores = mm(np.where(zh > f32(0), zh, f32(0)), w2) + b2
    return np.concatenate([a.ravel() for a in (w1, b1, w2, b2)]), scores, scores.argmax(1).astype(np.int64)
