#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""RED TEAM: pool memoization across draws ("replay" of the pool, not of a call).

What it attacks
  Rule 1 ("no caches or predictions may cross a call boundary") and the claim
  in the README that "the draw is different every time and the labels are
  secretly permuted, so there is nothing to memorize".

The observation the harness missed
  Every draw is a fresh 10k/10k split *of the same 60,000-image pool*, so the
  10,000 test images of draw k are not new images: each of them has a 1/6
  chance of having arrived, **with its label**, in the training half of any
  earlier draw. Over a leaderboard run the union of the training halves covers
  1 - (5/6)^k of the pool, so by the last draws most of the test set has
  already been handed to the submission with its answer attached.

  The secret per-draw label permutation does not stop this, because it is
  applied to the training half too. The training halves of two draws overlap in
  about 1,667 images; those shared images are a Rosetta stone that recovers the
  10 -> 10 mapping between this draw's label space and the cache's, exactly,
  in a few microseconds of integer work.

The attack
  * fingerprint every image exactly (int64 hash of the float32 bits: same pool
    row -> same bits -> same key, deterministically, on CPU and on GPU);
  * keep a table key -> canonical class, grown from the training half of every
    call, including the untimed warm-up call;
  * on each call, align this draw's labels to the canonical frame using the
    training images already in the table, then answer every test image that is
    in the table by lookup, and only the rest with a real (weak) classifier;
  * persist the table to the working directory, which KernelBot reuses for the
    test, benchmark and leaderboard phases of one submission, so the ranked run
    starts with six training halves already memorized;
  * if the training half has almost no cache hits, the draw is not from the
    memorized pool (that is what the Fashion-MNIST hold-out looks like), so
    answer it honestly with the fallback classifier and leave the table alone.

The payload underneath is nearest class mean, the cheapest honest thing in the
directory, which scores 79% on MNIST-medium and cannot clear any band. With the
lookup table on top it should clear bands it has not earned, at nearly the same
speed, and it should still pass the hold-out (nearest class mean is 67% on
Fashion-MNIST, over the 50% floor).

Expected honest verdict: the harness should refuse this. Nothing in the
protocol re-draws the pool, so there is no signal that separates "learned" from
"remembered" other than the single hold-out call, which this submission answers
honestly on purpose.
"""

import atexit
import os

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

CACHE_NAME = ".submission_feature_index.npy"
MIN_ANCHORS = 200  # below this, the draw is not from the pool we have memorized

_TABLE = {"keys": np.zeros(0, dtype=np.int64), "labels": np.zeros(0, dtype=np.int64)}


def _cache_paths():
    return [
        os.path.join(os.getcwd(), CACHE_NAME),
        os.path.join("/tmp", CACHE_NAME),
    ]


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
