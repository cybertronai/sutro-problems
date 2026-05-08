"""Sparse-parity scorer package — re-exports the public API from
``sparse_parity.sparse_parity``.

Lets ``import sparse_parity`` work from outside the
``sparse-parity/`` directory:

    from sparse_parity import score_small, generate_baseline_small
    cost = score_small(generate_baseline_small())
"""
from .sparse_parity import (  # noqa: F401
    Spec,
    SMALL,
    MEDIUM,
    N_BITS,
    K_SECRET,
    M_TRAIN,
    M_TEST,
    generate,
    solve_bruteforce,
    predict,
    accuracy,
    score_small,
    generate_baseline_small,
    score_medium,
    generate_baseline_medium,
)

# Re-export private helpers so the in-tree test suite can probe them.
from .sparse_parity import (  # noqa: F401
    _simulate, _cost, _parse, _instance,
    _CANONICAL_SEEDS, _CANONICAL_SEEDS_SMALL, _CANONICAL_SEEDS_MEDIUM,
)

__all__ = [
    "Spec", "SMALL", "MEDIUM",
    "N_BITS", "K_SECRET", "M_TRAIN", "M_TEST",
    "generate", "solve_bruteforce", "predict", "accuracy",
    "score_small", "generate_baseline_small",
    "score_medium", "generate_baseline_medium",
]
