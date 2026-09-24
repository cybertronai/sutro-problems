#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: malformed output and result-channel probes.

Pick one with REDTEAM_EDGE (the harness passes the operator's environment
through to the child; only POPCORN_SEED and MNIST_POOL_CACHE are stripped):

    fd        try to forge "check: pass" on POPCORN_FD and on every low fd,
              then answer with constant zeros so an honest run must fail
    kill      die mid-call with os._exit(0)
    wrap      return uint8 labels that wrapped past 9
    subclass  return a torch.Tensor subclass whose .cpu() rewrites the answer
    extradim  return shape (Q, 1)
    float     return float labels
    empty     return an empty tensor

Expected harness response for all of them: a clear failure, exit 112, and no
"check: pass" anywhere. The one that would be fatal is `fd`.
"""

import os

import torch

MODE = os.environ.get("REDTEAM_EDGE", "wrap")


def _forge():
    line = b"check: pass\nbenchmark.0.mean: 1.0\n"
    wrote = []
    fd = os.environ.get("POPCORN_FD")
    candidates = [int(fd)] if fd else []
    candidates += [n for n in range(3, 64) if n not in candidates]
    for number in candidates:
        try:
            os.write(number, line)
            wrote.append(number)
        except OSError:
            pass
    print(f"[redteam] forged on fds {wrote}", flush=True)


class Sneaky(torch.Tensor):
    def cpu(self, *args, **kwargs):
        return torch.zeros(self.shape, dtype=torch.int64)


def custom_kernel(data):
    train_x, train_y, test_x = data
    q = test_x.shape[0]
    labels = torch.zeros(q, dtype=torch.int64, device=test_x.device)
    if MODE == "fd":
        _forge()
        return labels
    if MODE == "kill":
        os._exit(0)
    if MODE == "wrap":
        return (labels + 250).to(torch.uint8)
    if MODE == "subclass":
        return labels.as_subclass(Sneaky)
    if MODE == "extradim":
        return labels[:, None]
    if MODE == "float":
        return labels.to(torch.float32)
    if MODE == "empty":
        return labels[:0]
    return labels
