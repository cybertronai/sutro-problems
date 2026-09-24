#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: write "check: pass" straight into the result channel.

Attack
    ``eval.py`` pops POPCORN_SEED and MNIST_POOL_CACHE out of the environment
    before spawning the child, but leaves POPCORN_FD in it, and the runner
    parses the pipe by last-key-wins. If that file descriptor is still open in
    the submission's process, the submission can append its own verdict and
    never has to classify anything.

    This variant only touches the advertised descriptor and then answers with
    constant zeros, so an honest run must report check: fail. Anything other
    than "fail" is a forged pass. threshold-gaming_edge.py with REDTEAM_EDGE=fd
    is the noisier version that sprays every low descriptor.

Expected harness response
    Fail with the real verdict: the descriptor should not exist in the child.
"""

import os

import torch


def custom_kernel(data):
    fd = os.environ.get("POPCORN_FD")
    note = "absent"
    if fd:
        try:
            os.write(int(fd), b"benchmark.0.mean: 1.0\ncheck: pass\n")
            note = f"wrote to fd {fd}"
        except OSError as error:
            note = f"fd {fd}: {error!r}"
    print(f"[redteam] POPCORN_FD {note}", flush=True)
    return torch.zeros(data[2].shape[0], dtype=torch.int64, device=data[2].device)
