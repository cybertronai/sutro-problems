"""Sparse-parity scorer package — re-exports the public API from
``sparse_parity.sparse_parity``.

Lets ``import sparse_parity`` work from outside the
``wip-sparse-parity/`` directory:

    from sparse_parity import score_sparse_parity, generate_baseline
    cost = score_sparse_parity(generate_baseline())
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
    score_sparse_parity,
    generate_baseline,
)

# Re-export private helpers so the in-tree test suite can probe them.
from .sparse_parity import _simulate, _cost, _parse, _sparse_parity_test  # noqa: F401

__all__ = [
    "N_BITS", "K_SECRET", "M_TRAIN", "M_TEST",
    "generate", "solve_bruteforce", "predict", "accuracy",
    "score_sparse_parity", "generate_baseline",
]
