"""Tests for symmetry.score_six / score_eight.  Run with:
``python3 -m pytest test_symmetry.py`` or ``python3 test_symmetry.py``."""
from __future__ import annotations

import pytest

import symmetry


def test_is_palindrome_known_cases():
    assert symmetry.is_palindrome([1, 0, 1, 1, 0, 1]) is True
    assert symmetry.is_palindrome([0, 0, 0, 0, 0, 0]) is True
    assert symmetry.is_palindrome([1, 1, 1, 1, 1, 1]) is True
    assert symmetry.is_palindrome([1, 0, 0, 0, 0, 1]) is True   # (0,5) match; (1,4) match; (2,3) match
    assert symmetry.is_palindrome([1, 0, 0, 0, 1, 0]) is False  # (0,5) differ
    assert symmetry.is_palindrome([0, 1, 0, 0, 0, 0]) is False  # (1,4) differ


def test_palindrome_count_six():
    """Exactly 8 of the 64 6-bit patterns are palindromes (2^3 free bits)."""
    count = sum(
        1 for code in range(64)
        if symmetry.is_palindrome([(code >> i) & 1 for i in range(6)])
    )
    assert count == 8


def test_palindrome_count_eight():
    """Exactly 16 of the 256 8-bit patterns are palindromes."""
    count = sum(
        1 for code in range(256)
        if symmetry.is_palindrome([(code >> i) & 1 for i in range(8)])
    )
    assert count == 16


def test_simulate_worked_example():
    """Single-instruction worked example: ``1;copy 2,1;2`` costs cost(1)+cost(2)=1+2=3."""
    actual, cost = symmetry._simulate("1;copy 2,1;2", [42])
    assert actual == [42]
    assert cost == 3


def test_set_op_is_free():
    """``set`` writes an immediate at zero instruction cost."""
    actual, cost = symmetry._simulate("1;set 2,7;2", [0])
    assert actual == [7]
    assert cost == symmetry._cost(2)  # only the output read


def test_baseline_six_correct_and_cost():
    """The 6-bit CMP+AND baseline passes all 64 patterns and costs exactly 20."""
    ir   = symmetry.generate_baseline_six()
    cost = symmetry.score_six(ir)
    assert cost == 20


def test_baseline_eight_correct_and_cost():
    """The 8-bit CMP+AND baseline passes all 256 patterns and costs exactly 29."""
    ir   = symmetry.generate_baseline_eight()
    cost = symmetry.score_eight(ir)
    assert cost == 29


def test_baseline_six_structure():
    """Baseline IR has exactly 6 inputs, 5 ops (3 cmp + 2 and), 1 output."""
    ir    = symmetry.generate_baseline_six()
    lines = [ln for ln in ir.splitlines() if ln.strip()]
    assert len(lines) == 7          # input + 3 cmp + 2 and + output
    assert lines[0] == "1,2,3,4,5,6"
    assert lines[1].startswith("cmp")
    assert lines[-1] == "1"


def test_score_rejects_wrong_output():
    """An IR that always outputs 0 fails on the all-zeros palindrome."""
    ir = "1,2,3,4,5,6\nset 7, 0\n7"
    with pytest.raises(ValueError, match="incorrect output"):
        symmetry.score_six(ir)


def test_score_rejects_wrong_input_count():
    """An IR with 5 inputs is rejected before testing patterns."""
    ir = symmetry.generate_baseline(n_bits=4)
    with pytest.raises(ValueError, match="inputs"):
        symmetry.score_six(ir)


def test_xor_not_baseline():
    """An alternative XOR+OR+sub implementation also passes all 64 patterns."""
    # xor-based: any_diff = (x0^x5)|(x1^x4)|(x2^x3); result = 1 - any_diff
    ir = "\n".join([
        "1,2,3,4,5,6",
        "xor 7, 1, 6",
        "xor 8, 2, 5",
        "xor 9, 3, 4",
        "or 7, 8",
        "or 7, 9",
        "set 8, 1",
        "sub 7, 8, 7",
        "7",
    ])
    cost = symmetry.score_six(ir)
    assert cost == 34


if __name__ == "__main__":
    tests = [
        test_is_palindrome_known_cases,
        test_palindrome_count_six,
        test_palindrome_count_eight,
        test_simulate_worked_example,
        test_set_op_is_free,
        test_baseline_six_correct_and_cost,
        test_baseline_eight_correct_and_cost,
        test_baseline_six_structure,
        test_score_rejects_wrong_output,
        test_score_rejects_wrong_input_count,
        test_xor_not_baseline,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
