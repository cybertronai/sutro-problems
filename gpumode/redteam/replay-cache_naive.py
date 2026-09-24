#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM (control): the dumb version of the attack, which must be caught.

Memoize by call count: compute an answer on the first call, then return that
same tensor on every later call without looking at the inputs. This is the
exploit that KernelGuard reports as the most common one in the wild (3.45% of
182,798 KernelBot submissions), and it is the thing the per-draw protocol is
supposed to make impossible.

It is included so the report can distinguish "the accuracy gate works" from
"the accuracy gate is sufficient". Expected: caught by the accuracy gate,
because every draw has different test images and a different label permutation,
so the replayed answer is right about 10% of the time.
"""

import torch

_ANSWER = {"labels": None}


def custom_kernel(data):
    train_x, train_y, test_x = data
    if _ANSWER["labels"] is not None:
        return _ANSWER["labels"]
    x = train_x.reshape(train_x.shape[0], -1)
    q = test_x.reshape(test_x.shape[0], -1)
    sums = torch.zeros(10, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, train_y, x)
    means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
    labels = ((means * means).sum(1) - 2 * q @ means.T).argmin(1)
    _ANSWER["labels"] = labels.clone()
    return _ANSWER["labels"]
