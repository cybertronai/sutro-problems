"""Tests for sparse_parity.score_small. Run with:
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
    """v3's ``set`` writes an integer immediate at zero cost."""
    actual, cost = sparse_parity._simulate("1;set 2,1;set 3,0;2,3", [42])
    assert actual == [1, 0]
    # Reads: addr 2 at exit (cost 2) + addr 3 at exit (cost 2). Both
    # `set` ops are free.
    assert cost == 4


def test_canonical_seeds_are_random_and_cover_all_secrets():
    """``_canonical_seeds`` returns random seeds covering every C(3,2)=3
    secret subset for the small instance, and produces a different seed
    list across calls (drawn from a fresh nondeterministic RNG)."""
    seeds_a = sparse_parity._canonical_seeds(sparse_parity.SMALL, max_seeds=64)
    seeds_b = sparse_parity._canonical_seeds(sparse_parity.SMALL, max_seeds=64)
    assert len(seeds_a) == 3 and len(seeds_b) == 3
    secrets_a = {
        tuple(sparse_parity.generate(seed=s, spec=sparse_parity.SMALL)[4])
        for s in seeds_a
    }
    assert secrets_a == {(0, 1), (0, 2), (1, 2)}
    # Two independent draws should almost never coincide on all three seeds.
    assert seeds_a != seeds_b, (seeds_a, seeds_b)


def test_baseline_small_robust_across_secrets():
    """The small baseline IR predicts y_test correctly on every
    canonical seed and reports cost 6918."""
    cost = sparse_parity.score_small(sparse_parity.generate_baseline_small())
    assert cost == 6918


def test_baseline_medium_robust_across_secrets():
    """The medium baseline IR predicts y_test correctly on every
    medium canonical seed and reports cost 816,251."""
    cost = sparse_parity.score_medium(sparse_parity.generate_baseline_medium())
    assert cost == 816251


def test_specialized_ir_rejected_by_robust_scorer():
    """A predictor IR specialized to a single hardcoded secret (here [1,2])
    must fail the multi-seed scorer because canonical seeds use different
    secrets."""
    M_TRAIN, M_TEST, N_BITS = (
        sparse_parity.M_TRAIN, sparse_parity.M_TEST, sparse_parity.N_BITS,
    )
    pred_base = 1
    X_tr_base = pred_base + M_TEST
    y_tr_base = X_tr_base + N_BITS * M_TRAIN
    X_te_base = y_tr_base + M_TRAIN
    inputs = list(range(X_tr_base, X_te_base + N_BITS * M_TEST))
    ops = [
        # secret = [1, 2] hardcoded (correct only for seeds whose secret is [1,2])
        f"xor {pred_base + j},"
        f"{X_te_base + j * N_BITS + 1},{X_te_base + j * N_BITS + 2}"
        for j in range(M_TEST)
    ]
    outputs = list(range(pred_base, pred_base + M_TEST))
    specialized = "\n".join(
        [",".join(map(str, inputs))] + ops + [",".join(map(str, outputs))])
    with pytest.raises(ValueError, match="correctness failed"):
        sparse_parity.score_small(specialized)


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
