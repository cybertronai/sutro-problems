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


def test_canonical_seeds_cover_all_secrets():
    """``_CANONICAL_SEEDS`` exercises every C(3,2)=3 secret subset."""
    secrets = set()
    for seed in sparse_parity._CANONICAL_SEEDS:
        _, _, _, _, secret = sparse_parity.generate(seed=seed)
        secrets.add(tuple(secret))
    assert len(secrets) == 3


def test_baseline_robust_across_secrets():
    """The general baseline IR predicts y_test correctly on every
    canonical seed and reports cost 1269."""
    cost = sparse_parity.score_sparse_parity(sparse_parity.generate_baseline())
    assert cost == 1269


def test_specialized_ir_rejected_by_robust_scorer():
    """A predictor IR specialized to seed=0 (secret=[1,2]) must fail
    the multi-seed scorer because secrets differ across canonical seeds."""
    specialized = "\n".join([
        ",".join(str(x) for x in range(1, 36)),
        "xor 36,22,23",  # X_test[0,1] ^ X_test[0,2] -- only correct for S=[1,2]
        "xor 37,25,26",
        "xor 38,28,29",
        "xor 39,31,32",
        "xor 40,34,35",
        "36,37,38,39,40",
    ])
    with pytest.raises(ValueError, match="correctness failed"):
        sparse_parity.score_sparse_parity(specialized)


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
