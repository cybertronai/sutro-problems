"""Sparse-parity scorer package wrapper."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("sparse_parity.py")
_SPEC = importlib.util.spec_from_file_location("_sparse_parity_impl", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"could not load sparse_parity implementation from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


Spec = _MODULE.Spec
SMALL = _MODULE.SMALL
MEDIUM = _MODULE.MEDIUM
N_BITS = _MODULE.N_BITS
K_SECRET = _MODULE.K_SECRET
M_TRAIN = _MODULE.M_TRAIN
M_TEST = _MODULE.M_TEST
generate = _MODULE.generate
solve_bruteforce = _MODULE.solve_bruteforce
predict = _MODULE.predict
accuracy = _MODULE.accuracy
score_small = _MODULE.score_small
generate_baseline_small = _MODULE.generate_baseline_small
score_medium = _MODULE.score_medium
generate_baseline_medium = _MODULE.generate_baseline_medium

# Re-export private helpers so the in-tree test suite can probe them.
_simulate = _MODULE._simulate
_cost = _MODULE._cost
_instance = _MODULE._instance
_canonical_seeds = _MODULE._canonical_seeds


__all__ = [
    "Spec", "SMALL", "MEDIUM",
    "N_BITS", "K_SECRET", "M_TRAIN", "M_TEST",
    "generate", "solve_bruteforce", "predict", "accuracy",
    "score_small", "generate_baseline_small",
    "score_medium", "generate_baseline_medium",
]
