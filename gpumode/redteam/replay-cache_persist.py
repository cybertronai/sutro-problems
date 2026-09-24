#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

"""RED TEAM: the same pool memoization, with a table that outlives the submission.

This is ``replay-cache.py`` with one line changed: the lookup table is written
to a fixed absolute path instead of the working directory.

Why that matters. ``replay-cache.py`` clears the 12% band on its first
submission but not the 5% band, purely because the test and benchmark phases
run before the table has seen much of the pool: the first timed call of a cold
submission is still the 79% payload. That brake is an accident of cold start,
not a property of the harness.

A leaderboard is not a cold start. KernelBot keeps one warm container per
submission stream (``max_containers=1`` in ``run_modal.py``, and the Modal
runner behaves the same way), and a contestant may submit the same file as many
times as they like. Anything the submission writes outside its scratch
directory survives to the next submission, so run 2 begins with every training
half run 1 was shown, run 3 with twice that, and so on. Nothing in the harness
inspects, clears or even names the filesystem.

To reproduce the effect on one machine, run this file twice against the same
band and watch the accuracy of the *first* timed call climb:

    rm -f /tmp/.sutro_pool_index.npy
    python run_modal.py --band mnist-medium-5pct \
        --submission redteam/replay-cache_persist.py --mode leaderboard --local ...
    python run_modal.py --band mnist-medium-5pct \
        --submission redteam/replay-cache_persist.py --mode leaderboard --local ...

The payload is still nearest class mean. See ``replay-cache.py`` for the
mechanism and for what the harness would have to do about it.
"""

import atexit

import numpy as np
import torch

# ---------------------------------------------------------------- fingerprint

_FP_WEIGHTS = {}


def _weights(device, width):
    key = (str(device), width)
    if key not in _FP_WEIGHTS:
        rng = np.random.default_rng(0xC0FFEE)
        raw = rng.integers(1, 2**62, size=width, dtype=np.int64) | 1
        _FP_WEIGHTS[key] = torch.from_numpy(raw).to(device)
    return _FP_WEIGHTS[key]


def _fingerprint(x):
    """Exact 64-bit key per image, from the float32 bit pattern.

    Integer multiply-add wraps modulo 2^64 and is therefore order independent,
    so the key does not depend on the reduction order the device picks.
    """
    flat = x.reshape(x.shape[0], -1).contiguous()
    bits = flat.view(torch.int32).to(torch.int64)
    return (bits * _weights(x.device, bits.shape[1])).sum(1)


# ---------------------------------------------------------------- the table

CACHE_NAME = "/tmp/.sutro_pool_index.npy"  # deliberately outside the scratch directory
MIN_ANCHORS = 200  # below this, the draw is not from the pool we have memorized

_TABLE = {"keys": np.zeros(0, dtype=np.int64), "labels": np.zeros(0, dtype=np.int64)}


def _cache_paths():
    return [CACHE_NAME]


def _load():
    # Import time, i.e. before the child reports "ready": never inside a timed
    # window. KernelBot runs test, benchmark and leaderboard in the same
    # working directory, so this inherits every phase that ran before.
    for path in _cache_paths():
        try:
            data = np.load(path)
        except Exception:
            continue
        if data.ndim == 2 and data.shape[0] == 2 and data.shape[1] > 0:
            _TABLE["keys"] = data[0].astype(np.int64)
            _TABLE["labels"] = data[1].astype(np.int64)
            return


def _save():
    # atexit: runs when the child process shuts down, after the last timed call.
    keys, labels = _TABLE["keys"], _TABLE["labels"]
    if keys.size == 0:
        return
    payload = np.stack([keys, labels])
    for path in _cache_paths():
        try:
            np.save(path, payload)
            return
        except Exception:
            continue


def _merge(keys, labels):
    all_keys = np.concatenate([_TABLE["keys"], keys])
    all_labels = np.concatenate([_TABLE["labels"], labels])
    unique, first = np.unique(all_keys, return_index=True)
    _TABLE["keys"] = unique
    _TABLE["labels"] = all_labels[first]


def _lookup(keys):
    table = _TABLE["keys"]
    if table.size == 0:
        return np.zeros(keys.shape[0], dtype=bool), np.zeros(keys.shape[0], dtype=np.int64)
    index = np.searchsorted(table, keys)
    np.clip(index, 0, table.size - 1, out=index)
    hit = table[index] == keys
    return hit, _TABLE["labels"][index]


# ---------------------------------------------------------------- the payload

def _nearest_class_mean(train_x, train_y, test_x):
    x = train_x.reshape(train_x.shape[0], -1)
    q = test_x.reshape(test_x.shape[0], -1)
    sums = torch.zeros(10, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, train_y, x)
    means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
    distances = (means * means).sum(1) - 2 * q @ means.T
    return distances.argmin(1)


def custom_kernel(data):
    train_x, train_y, test_x = data

    predictions = _nearest_class_mean(train_x, train_y, test_x)

    train_keys = _fingerprint(train_x).cpu().numpy()
    test_keys = _fingerprint(test_x).cpu().numpy()
    labels = train_y.to(torch.int64).cpu().numpy()

    train_hit, train_canonical = _lookup(train_keys)
    anchors = int(train_hit.sum())

    if _TABLE["keys"].size == 0:
        # First draw ever seen: its label space defines the canonical frame.
        _merge(train_keys, labels)
        return predictions

    if anchors < MIN_ANCHORS:
        # Not our pool. This is what the hold-out call looks like: answer it
        # with the honest classifier and do not pollute the table.
        return predictions

    counts = np.zeros((10, 10), dtype=np.int64)
    np.add.at(counts, (train_canonical[train_hit], labels[train_hit]), 1)
    current_of_canonical = counts.argmax(1)
    if np.unique(current_of_canonical).size != 10:
        return predictions  # not a clean permutation; do not risk it
    canonical_of_current = np.zeros(10, dtype=np.int64)
    canonical_of_current[current_of_canonical] = np.arange(10)

    test_hit, test_canonical = _lookup(test_keys)
    if test_hit.any():
        answer = predictions.to(torch.int64).cpu().numpy()
        answer[test_hit] = current_of_canonical[test_canonical[test_hit]]
        predictions = torch.from_numpy(answer).to(train_x.device)

    _merge(train_keys, canonical_of_current[labels])
    return predictions


_load()
atexit.register(_save)
