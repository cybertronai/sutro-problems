"""Matmul scorer package — re-exports the public API from ``matmul.matmul``.

Lets ``import matmul`` work from outside the ``matmul/`` directory:

    from matmul import score_4x4, generate_baseline_4x4
    cost = score_4x4(generate_baseline_4x4())
"""
from .matmul import (  # noqa: F401
    score_1x1,
    score_4x4,
    score_16x16,
    generate_baseline_4x4,
    generate_baseline_16x16,
    generate_tiled_16x16,
)

# Re-export private helpers so the in-tree test suite can probe them.
from .matmul import _simulate, _cost, _parse, _matmul_test  # noqa: F401

__all__ = [
    "score_1x1",
    "score_4x4",
    "score_16x16",
    "generate_baseline_4x4",
    "generate_baseline_16x16",
    "generate_tiled_16x16",
]
