#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""Nearest class mean: one index_add, one 10x81 matmul, one argmin.

About 80% accurate on MNIST-medium, so it fails every band including 12%. It is
the fastest honest thing in this directory and exists to exercise the harness
end to end and to give the accuracy gate something that must be rejected.

Identical to reference.py and to the body shipped in submission.py.
"""

import torch


def custom_kernel(data):
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], -1)
    q = test_x.reshape(test_x.shape[0], -1)
    sums = torch.zeros(10, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, train_y, x)
    means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
    distances = (means * means).sum(1) - 2 * q @ means.T
    return distances.argmin(1)
