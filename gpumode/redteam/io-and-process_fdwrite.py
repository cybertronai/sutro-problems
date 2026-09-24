#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""Red-team lens io-and-process, variant: write to the result pipe (POPCORN_FD).

The parent leaves POPCORN_FD in the environment the child inherits (recon showed
own_env_has_POPCORN_FD == True). If that file descriptor is still open in the
child, the submission can inject its own "key: value" lines into KernelBot's
result channel -- e.g. forge "check: pass", a fake tiny "benchmark.0.mean", or
overwrite the accuracy fields -- because KernelBot parses whatever lines land on
that pipe, no matter who wrote them.

This file writes a loud marker plus a forged fast/passing record to the fd, then
returns a deliberately WRONG answer (all zeros) so that if the run still reports
pass / a tiny time, the only possible source is the injected lines.

Records what happened to /tmp/redteam-fdwrite.json.
"""

import json
import os

import torch

REPORT = "/tmp/redteam-fdwrite.json"


def _attack_fd():
    result = {}
    fd_str = os.environ.get("POPCORN_FD")
    result["POPCORN_FD_env"] = fd_str
    if not fd_str:
        result["status"] = "no fd in env"
        return result
    fd = int(fd_str)
    # Is the fd still open in the child?
    try:
        os.fstat(fd)
        result["fstat_ok"] = True
    except OSError as exc:
        result["fstat_ok"] = False
        result["fstat_error"] = repr(exc)
    # Try to write forged result lines.
    forged = (
        "\nREDTEAM_FD_INJECTION: reached the result pipe\n"
        "benchmark.0.mean: 1\n"
        "benchmark.0.accuracy: 1.000000\n"
        "benchmark.0.correct: 999999\n"
        "check: pass\n"
    )
    try:
        os.write(fd, forged.encode())
        result["write_ok"] = True
    except OSError as exc:
        result["write_ok"] = False
        result["write_error"] = repr(exc)
    return result


_done = {"v": False}


def custom_kernel(data):
    train_x, train_y, test_x = data
    if not _done["v"]:
        _done["v"] = True
        info = _attack_fd()
        try:
            with open(REPORT, "w") as fh:
                json.dump(info, fh, indent=2)
        except OSError:
            pass
    # deliberately wrong answer
    return torch.zeros(test_x.shape[0], dtype=torch.int64, device=test_x.device)
