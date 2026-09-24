#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

"""512-unit MLP, ported from mnist/submissions/medium-affine-20260911.

That submission is a NumPy learner whose arithmetic is pinned to a spatial-grid
IR (ordered FP32 reductions, einsum with optimize=False). This file is a plain
PyTorch transcription of the same architecture, the same update rule and the
same constants, not a bit-for-bit copy of its reduction order; the original is
documented in that directory's README.md and config.json:

  input transform   x * 4 - 0.5 on the flattened 81 pixels
  architecture      81 -> 512 ReLU -> 10, biases start at zero
  initialization    numpy PCG64(101), uniform(-1/sqrt(81), 1/sqrt(81)) for the
                    first layer and uniform(-1/sqrt(512), 1/sqrt(512)) for the
                    second
  loss              squared error against one-hot targets, gradient scaled by
                    the learning rate divided by the minibatch size
  optimizer         plain SGD, learning rate 0.1
  schedule          200 epochs of contiguous minibatches of 30, in data order

The original reported 96% on the 6,000/6,000 medium tier. The upstream learner
asserts that the training count divides 30; with 10,000 examples the last
minibatch of each epoch holds 10, which changes nothing else.

The initial parameters are a seeded constant. They are cloned at the start of
every call, so each call trains from scratch.

Harness 1.2.0 release. The harness now hands over (N, D) features -- the
draw's images whitened onto D = 60 principal directions and secretly rotated --
instead of (N, 1, 9, 9) pixels. Two lines change: the input width is read off
the tensor instead of the constant 81 (so the seeded initialization is
uniform(-1/sqrt(D), 1/sqrt(D)) at whatever D arrives), and the pixel rescaling
``x * 4 - 0.5`` becomes a plain ``x * RELEASE_SCALE`` on a linear release.
Passing the release through unscaled looked right -- it is already zero-mean
and unit-variance -- but it is not: whitening spreads the same energy over all
60 directions instead of the ~18 the pixel scaling left, and at the fixed
lr = 0.1 the squared-error update then diverges on some draws (1 of 5 at
N = 2,000 and 2 of 11 at N = 10,000 collapsed onto a single class). Everything
else -- architecture, loss, SGD, schedule -- is untouched.
"""

import torch

WIDTH = 512
EPOCHS = 200
LEARNING_RATE = 0.1
BATCH = 30
SEED = 101
# The pixel branch fed x * 4 - 0.5, whose second moment is concentrated in about
# 18 informative directions; the linear release spreads unit variance over all
# 60, which puts the fixed lr = 0.1 squared-error update outside its stability
# limit on some draws (measured: a single-class collapse on 2 of 11 draws at
# N = 10,000). Scaling the release down restores the margin.
RELEASE_SCALE = 0.25

_initial = {}


def features_of(tensor):
    """Flatten to (N, D) and scale the input for the release it came from.

    A pixel release arrives as (N, 1, size, size) in [0, 1] and gets the upstream
    ``x * 4 - 0.5``; the linear release of harness 1.2.0 arrives already flat as
    (N, D) with unit variance per coordinate and gets ``RELEASE_SCALE``, which
    keeps the fixed-learning-rate update stable (see the constant above).
    """
    flat = tensor.reshape(tensor.shape[0], -1)
    return flat * 4.0 - 0.5 if tensor.dim() == 4 else flat * RELEASE_SCALE


def initial_parameters(device, features):
    """numpy PCG64(101) uniform initialization, materialized once per device and width."""
    key = (str(device), features)
    if key not in _initial:
        import math

        import numpy as np

        rng = np.random.Generator(np.random.PCG64(SEED))
        w1 = rng.uniform(
            -1 / math.sqrt(features), 1 / math.sqrt(features), (features, WIDTH)
        ).astype("float32")
        w2 = rng.uniform(-1 / math.sqrt(WIDTH), 1 / math.sqrt(WIDTH), (WIDTH, 10)).astype(
            "float32"
        )
        _initial[key] = (
            torch.as_tensor(w1, device=device),
            torch.zeros(WIDTH, device=device),
            torch.as_tensor(w2, device=device),
            torch.zeros(10, device=device),
        )
    return _initial[key]


@torch.no_grad()
def custom_kernel(data):
    train_x, train_y, test_x = data
    device = train_x.device
    x = features_of(train_x)
    q = features_of(test_x)
    target = torch.nn.functional.one_hot(train_y, 10).to(torch.float32)

    w1, b1, w2, b2 = (tensor.clone() for tensor in initial_parameters(device, x.shape[1]))

    count = x.shape[0]
    for _ in range(EPOCHS):
        for start in range(0, count, BATCH):
            xb = x[start : start + BATCH]
            tb = target[start : start + BATCH]
            z = xb @ w1 + b1
            h = torch.relu(z)
            scores = h @ w2 + b2
            d2 = scores - tb
            d1 = torch.where(z > 0, d2 @ w2.T, torch.zeros((), device=device))
            step = LEARNING_RATE / xb.shape[0]
            w2 -= step * (h.T @ d2)
            b2 -= step * d2.sum(0)
            w1 -= step * (xb.T @ d1)
            b1 -= step * d1.sum(0)

    return (torch.relu(q @ w1 + b1) @ w2 + b2).argmax(1)
