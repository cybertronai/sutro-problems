"""Tests for sparse_parity.score_sparse_parity. Run with:
``python3 -m pytest test_sparse_parity.py`` or just
``python3 test_sparse_parity.py`` (built-in __main__ runner)."""
from __future__ import annotations

import pytest

import sparse_parity


def test_score_worked_example():
    """Documented one-liner: ``1,2;xor 3,1,2;3`` → cost 5.

    Reads:
      * ``xor 3,1,2``  reads addr 1 (cost ⌈√1⌉=1) + addr 2 (cost ⌈√2⌉=2)
      * exit           reads addr 3 (cost ⌈√3⌉=2)
    Total: 1 + 2 + 2 = 5.
    """
    actual, cost = sparse_parity._simulate("1,2;xor 3,1,2;3", [1, 0])
    assert actual == [1]
    assert cost == 5


def test_set_op_is_free():
    """v2's ``set`` writes an integer immediate at zero cost."""
    actual, cost = sparse_parity._simulate("1;set 2,1;set 3,0;2,3", [42])
    assert actual == [1, 0]
    # Reads: addr 2 at exit (cost 2) + addr 3 at exit (cost 2). Both
    # `set` ops are free.
    assert cost == 4


def test_baseline_recovers_secret():
    """The naive baseline IR predicts y_test exactly and reports cost 91."""
    cost = sparse_parity.score_sparse_parity(sparse_parity.generate_baseline())
    assert cost == 91


def test_python_reference_solver():
    """``solve_bruteforce`` recovers the secret across many seeds."""
    for seed in range(50):
        Xtr, ytr, Xte, yte, secret = sparse_parity.generate(seed=seed)
        found = sparse_parity.solve_bruteforce(Xtr, ytr)
        assert found == secret
        pred = sparse_parity.predict(Xte, found)
        assert sparse_parity.accuracy(pred, yte) == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
