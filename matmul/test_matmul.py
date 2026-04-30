"""Tests for matmul.score_*. Run with: ``python3 -m pytest test_matmul.py``
or just ``python3 test_matmul.py`` (built-in __main__ runner)."""
from __future__ import annotations

import math
import pytest

import matmul


# ---------------------------------------------------------------------------
# Worked examples from README
# ---------------------------------------------------------------------------

def test_score_1x1_worked_example():
    """Documented one-liner: `1,2;mul 3,1,2;3` → cost 5.

    Reads:
      * `mul 3,1,2`  reads addr 1 (cost ⌈√1⌉=1) + addr 2 (cost ⌈√2⌉=2)
      * exit         reads addr 3 (cost ⌈√3⌉=2)
    Total: 1 + 2 + 2 = 5.
    """
    assert matmul.score_1x1("1,2;mul 3,1,2;3") == 5


def test_score_myfunc_worked_example():
    """Documented `myfunc(a,b,c,d,e) = a*b + c*d + e` example: total cost 15."""
    ir = (
        "1,2,3,4,5\n"
        "mul 1,1,2\n"   # t1 = a*b → addr 1   (1+2)
        "mul 2,3,4\n"   # t2 = c*d → addr 2   (2+2)
        "add 1,1,2\n"   # s  = t1+t2 → addr 1 (1+2)
        "add 1,5\n"     # r  = s+e → addr 1   (1+3)  in-place short form
        "1\n"           # exit                (1)
    )
    # Not a matmul, so we drive _simulate directly with the 5 inputs.
    outputs, cost = matmul._simulate(ir, [10, 20, 30, 40, 50])
    assert cost == 15
    assert outputs == [10 * 20 + 30 * 40 + 50]


# ---------------------------------------------------------------------------
# Baselines run to completion + match published costs
# ---------------------------------------------------------------------------

def test_baseline_4x4_cost_matches_record_history():
    assert matmul.score_4x4(matmul.generate_baseline_4x4()) == 1_316


def test_baseline_16x16_cost_matches_record_history():
    assert matmul.score_16x16(matmul.generate_baseline_16x16()) == 340_704


def test_tiled_16x16_cost_matches_record_history():
    assert matmul.score_16x16(matmul.generate_tiled_16x16()) == 133_783


# ---------------------------------------------------------------------------
# Newline / semicolon line separators are interchangeable
# ---------------------------------------------------------------------------

def test_newline_and_semicolon_give_same_cost():
    semi = "1,2;mul 3,1,2;3"
    nl   = "1,2\nmul 3,1,2\n3"
    assert matmul.score_1x1(semi) == matmul.score_1x1(nl)


# ---------------------------------------------------------------------------
# Correctness rejection — wrong arithmetic is caught
# ---------------------------------------------------------------------------

def test_score_1x1_rejects_wrong_arithmetic():
    """A 1×1 IR that ADDs instead of MULs gets the wrong answer; the
    scorer must raise ValueError, not silently return a cost."""
    bad = "1,2;add 3,1,2;3"
    with pytest.raises(ValueError, match="correctness failed"):
        matmul.score_1x1(bad)


# ---------------------------------------------------------------------------
# Cost-formula sanity for the worked-example reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("addr,expected", [
    (1, 1), (2, 2), (3, 2), (4, 2),
    (5, 3), (9, 3), (10, 4), (16, 4), (17, 5),
])
def test_cost_helper_ceil_sqrt(addr, expected):
    """⌈√addr⌉ should match math.isqrt(addr-1) + 1."""
    assert matmul._cost(addr) == expected
    # Cross-check against floating-point ceil(sqrt) for sanity.
    assert matmul._cost(addr) == math.ceil(math.sqrt(addr))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
