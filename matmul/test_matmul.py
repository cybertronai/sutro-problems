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

    The 1×1 test data is now ``A=[[1]], B=[[3]], C=[[3]]`` (B is no
    longer the transpose of A), so this IR — which actually computes
    ``A·B`` — gives the right answer 3, while a degenerate IR that
    just returned ``A[0][0]`` (= 1) would now fail correctness.
    """
    assert matmul.score_1x1("1,2;mul 3,1,2;3") == 5


def test_score_1x1_rejects_identity_ir():
    """Sanity check that the non-symmetric test data catches an
    "identity" IR that returns A[0][0] instead of A[0][0] * B[0][0].
    Under the old A=B^T data this passed coincidentally; under the
    new data ``A=[[1]], B=[[3]]`` it must fail correctness."""
    identity = "1,2;copy 3,1;3"   # mem[3] = mem[1] = A[0][0]; expects C != A
    with pytest.raises(ValueError, match="correctness failed"):
        matmul.score_1x1(identity)


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


def test_dead_input_outputs_packed_16x16_cost_matches_record_history():
    from matmul.submissions.dead_input_outputs_packed_16x16 import (
        generate_dead_input_outputs_packed_16x16,
    )

    assert matmul.score_16x16(
        generate_dead_input_outputs_packed_16x16()) == 70_053


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
# v0 instruction-set conformance
# ---------------------------------------------------------------------------

def test_legacy_mov_is_rejected():
    """v0 spells the data-movement op `copy`; legacy `mov` must raise."""
    with pytest.raises(ValueError, match="unknown op.*mov"):
        matmul.score_1x1("1,2;mov 3,1;3")


def test_unknown_op_is_rejected():
    with pytest.raises(ValueError, match="unknown op"):
        matmul.score_1x1("1,2;fma 3,1,2;3")


def test_2x2_non_symmetric_test_data():
    """Validation matrices for ``n=2`` produce a non-symmetric C, so an
    IR that confused i,j somewhere can't pass coincidentally."""
    inputs, expected = matmul._matmul_test(2)
    # A = [[1,2],[3,4]] (1..4),  B[i][j] = i + 2j + 3 → [[3,5],[4,6]],
    # C = A @ B = [[1·3+2·4, 1·5+2·6], [3·3+4·4, 3·5+4·6]]
    #            = [[11, 17], [25, 39]]
    assert expected == [11, 17, 25, 39]
    assert expected[1] != expected[2]   # not symmetric


# ---------------------------------------------------------------------------
# Positive-address validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_ir", [
    "0,2;mul 3,1,2;3",            # input addr 0
    "1,-1;mul 3,1,2;3",           # input addr -1
    "1,2;mul 0,1,2;3",            # op writing to addr 0
    "1,2;mul 3,0,2;3",            # op reading from addr 0
    "1,2;mul 3,1,2;0",            # output addr 0
    "1,2;copy -5,1;3",            # negative copy dest
])
def test_non_positive_address_raises(bad_ir):
    with pytest.raises(ValueError, match="positive integers"):
        matmul.score_1x1(bad_ir)


def test_cost_helper_rejects_zero():
    with pytest.raises(ValueError, match="positive integers"):
        matmul._cost(0)


def test_cost_helper_rejects_negative():
    with pytest.raises(ValueError, match="positive integers"):
        matmul._cost(-3)


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
