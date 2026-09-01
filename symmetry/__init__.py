"""Symmetry scorer package — re-exports the public API from ``symmetry.symmetry``.

Lets ``import symmetry`` work from outside the ``symmetry/`` directory:

    from symmetry import score_six, generate_baseline_six
    cost = score_six(generate_baseline_six())
"""
from .symmetry import (  # noqa: F401
    is_palindrome,
    score,
    score_six,
    score_eight,
    generate_baseline,
    generate_baseline_six,
    generate_baseline_eight,
)

# Re-export private helpers so the in-tree test suite can probe them.
from .symmetry import _simulate, _cost  # noqa: F401

__all__ = [
    "is_palindrome",
    "score",
    "score_six",
    "score_eight",
    "generate_baseline",
    "generate_baseline_six",
    "generate_baseline_eight",
    "_simulate",
    "_cost",
]
