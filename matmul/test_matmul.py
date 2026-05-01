"""Tests for matmul.score_*. Run with: ``python3 -m pytest test_matmul.py``
or just ``python3 test_matmul.py`` (built-in __main__ runner)."""
from __future__ import annotations

import pytest

import matmul


def test_score_1x1_worked_example():
    """Documented one-liner: `1,2;mul 3,1,2;3` → cost 5.

    Reads:
      * `mul 3,1,2`  reads addr 1 (cost ⌈√1⌉=1) + addr 2 (cost ⌈√2⌉=2)
      * exit         reads addr 3 (cost ⌈√3⌉=2)
    Total: 1 + 2 + 2 = 5.

    The 1×1 test data is now ``A=[[1]], B=[[3]], C=[[3]]`` (B is no
    longer the transpose of A), so this IR — which actually computes
    ``A·B`` — gives the right answer 3, while a degenerate IR that
    just returned ``A[0][0]`` (= 1) would now fail correctness.
    """
    assert matmul.score_1x1("1,2;mul 3,1,2;3") == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
