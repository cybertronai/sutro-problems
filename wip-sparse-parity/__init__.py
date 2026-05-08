"""Sparse-parity scorer package — re-exports the public API from
``sparse_parity.sparse_parity``.

Lets ``import sparse_parity`` work from outside the
``wip-sparse-parity/`` directory:

    from sparse_parity import score_small, generate_baseline_small
    cost = score_small(generate_baseline_small())
"""
from .sparse_parity import (  # noqa: F401
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
)

# Re-export private helpers so the in-tree test suite can probe them.
from .sparse_parity import (  # noqa: F401
    _simulate, _cost, _parse, _instance, _CANONICAL_SEEDS,
)

__all__ = [
    "N_BITS", "K_SECRET", "M_TRAIN", "M_TEST",
    "generate", "solve_bruteforce", "predict", "accuracy",
    "score_small", "generate_baseline_small",
]
