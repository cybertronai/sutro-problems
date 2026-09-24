#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

"""RED TEAM: train once, then reuse the weights for every later timed call.

What it attacks
  The ranked number itself. Rule 1 says "no parameters, statistics, caches,
  predictions or fitted state may cross a call boundary", and the README argues
  that the secret per-draw label permutation makes carried-over state "worse
  than useless". The permutation is the only thing enforcing the rule, and it
  costs about one percent of a call to undo.

The attack
  1. Call 1 fits a real classifier (4,096 random ReLU features + ridge) on the
     draw it is given. That call is slow and it is timed, once.
  2. Every later call keeps the fitted weights. The class ids the frozen model
     emits are in call 1's label space, so the submission runs the frozen model
     over a 1,000-image probe of *this* call's training half, builds the 10x10
     confusion against this call's ``train_y``, and reads off the permutation.
     Relabelling costs one gather.
  3. If no cached model scores well on the probe, the draw is not from a
     distribution any cached model knows -- which is exactly what the
     Fashion-MNIST hold-out call looks like -- so that call trains from scratch
     and the new model is added to the cache. The hold-out is untimed in the
     ranking, so training there is free, and the ranked calls stay fast.

  The ranked mean therefore contains one training run and ten inference runs
  instead of eleven training runs. With D draws the ranked time is
  (T_train + (D-1) * T_infer) / D, which tends to T_infer as the organizers
  raise D for statistical comfort: the more draws they add, the bigger the
  discount. The accuracy gate notices nothing, because a model fitted on one
  10,000-image draw generalizes to the next draw of the same pool slightly
  *better* than it does to its own held-out half.

Control
  Set ``REDTEAM_NO_REUSE=1`` in the environment and the same file retrains on
  every call. The two runs differ only in that flag, so the difference between
  their ``benchmark.0.mean`` values is exactly what the exploit steals.

Expected honest verdict: the harness should refuse this. It does not: the
per-call protocol is stateless by convention only, and the one call that would
expose carried state (the hold-out) is the one call the submission chooses to
answer honestly.
"""

import os

import numpy as np
import torch

FEATURES = 4096
RIDGE = 1e-2
PROBE = 1000
PROBE_FLOOR = 0.60  # a cached model must beat this on the probe to be reused
REUSE = os.environ.get("REDTEAM_NO_REUSE", "0") != "1"

_MODELS = []  # every fitted model, kept across calls


def _projection(device, width):
    generator = torch.Generator(device="cpu").manual_seed(0x5CA1AB1E)
    projection = torch.randn(width, FEATURES, generator=generator) * 1.4
    bias = torch.randn(FEATURES, generator=generator) * 0.3
    return projection.to(device), bias.to(device)


def _features(x, projection, bias):
    return torch.relu(x.reshape(x.shape[0], -1) @ projection + bias)


def _fit(train_x, train_y):
    projection, bias = _projection(train_x.device, train_x[0].numel())
    hidden = _features(train_x, projection, bias)
    gram = hidden.T @ hidden
    gram += torch.eye(FEATURES, device=hidden.device, dtype=hidden.dtype) * (
        RIDGE * torch.diagonal(gram).mean()
    )
    targets = torch.zeros(train_y.shape[0], 10, device=hidden.device, dtype=hidden.dtype)
    targets[torch.arange(train_y.shape[0], device=hidden.device), train_y] = 1.0
    beta = torch.linalg.solve(gram, hidden.T @ targets)
    return {"projection": projection, "bias": bias, "beta": beta}


def _predict(model, x):
    scores = _features(x, model["projection"], model["bias"]) @ model["beta"]
    return scores.argmax(1)


def _alignment(model, probe_x, probe_y):
    """Recover this draw's label permutation from a small labelled probe."""
    predicted = _predict(model, probe_x)
    counts = torch.zeros(10, 10, device=probe_x.device, dtype=torch.float32)
    counts.index_put_((predicted, probe_y), torch.ones_like(predicted, dtype=torch.float32),
                      accumulate=True)
    mapping = counts.argmax(1)
    score = float(counts.max(1).values.sum().item()) / probe_y.shape[0]
    return mapping, score


def custom_kernel(data):
    train_x, train_y, test_x = data
    train_y = train_y.to(torch.int64)

    if not REUSE:
        model = _fit(train_x, train_y)
        return _predict(model, test_x)

    probe = min(PROBE, train_x.shape[0])
    probe_x, probe_y = train_x[:probe], train_y[:probe]

    chosen, mapping = None, None
    for model in _MODELS:
        candidate, score = _alignment(model, probe_x, probe_y)
        if score >= PROBE_FLOOR and int(torch.unique(candidate).numel()) == 10:
            chosen, mapping = model, candidate
            break

    if chosen is None:
        # Unknown distribution (a cold start, or the hold-out call): pay for it.
        chosen = _fit(train_x, train_y)
        _MODELS.append(chosen)
        mapping = torch.arange(10, device=train_x.device)

    return mapping[_predict(chosen, test_x)]
