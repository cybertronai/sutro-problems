"""Baseline learner: nearest class mean. About 80% on MNIST-medium, so it misses every error band.

Learners receive the same three input tensors on every call (only their contents
change) and must train from scratch on each call: no state may carry over.
Class labels are secretly permuted per draw, so labels only mean something
relative to the training set they arrive with.
"""
import torch

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], -1)
    q = test_x.reshape(test_x.shape[0], -1)
    sums = torch.zeros(10, x.shape[1], device=x.device).index_add_(0, train_y, x)
    means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
    distances = (means * means).sum(1) - 2 * q @ means.T
    return distances.argmin(1)
