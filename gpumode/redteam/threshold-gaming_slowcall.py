#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: max_call_ms is a post-mortem, not a limit, and the warm-up has none.

Attack
    ``eval.py`` compares ``device_ms`` with ``max_call_ms`` only after
    ``child.call("timed", ...)`` has returned, so the limit cannot stop a call:
    it can only complain about one that already finished. A call that never
    finishes is never measured at all, and the evaluator's blocking
    ``connection.recv()`` waits for it until KernelBot's mode timeout (1200 s
    on a ranked run) expires. The warm-up call is not checked against
    ``max_call_ms`` at all, which is what makes an expensive pre-training pass
    (threshold-gaming.py) free.

    REDTEAM_SLOW_S seconds of sleeping per call, default 6, with
    ``--case max_call_ms=1000``: the harness still waits the full 6 s.

Expected harness response
    Fail, and ideally fail *at* the limit rather than after it.
"""

import os
import time

import torch

SLEEP_S = float(os.environ.get("REDTEAM_SLOW_S", "6"))


def custom_kernel(data):
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], -1)
    q = test_x.reshape(test_x.shape[0], -1)
    time.sleep(SLEEP_S)
    sums = torch.zeros(10, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, train_y, x)
    means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
    return ((means * means).sum(1) - 2 * q @ means.T).argmin(1)
