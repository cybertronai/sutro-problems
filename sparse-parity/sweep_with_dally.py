"""Candidate sweep scored by the dally-eval Rust engine.

Demonstrates the in-repo inner loop: generate candidate layouts of a
sparse-parity program, score each with dally-eval (via dally_eval.py),
and fall back to the Python evaluator only if the binary is missing.
Run from this directory:

    python3 sweep_with_dally.py            # ~200 candidates
    DALLY_EVAL_CANDS=2000 python3 sweep_with_dally.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import mask_sparse_parity as mp
import dally_eval


def candidates(n: int):
    """(seed, cap) grid over the siswalk family."""
    for seed in range(n // 3 + 1):
        for cap in (2, 3, 4):
            yield seed, cap


def main() -> None:
    n = int(os.environ.get("DALLY_EVAL_CANDS", "200"))
    engine = "dally-eval (rust)" if dally_eval.available() else "python fallback"
    print(f"engine: {engine}; candidates: ~{n}")

    t0 = time.perf_counter()
    results = []
    for seed, cap in candidates(n):
        ir = mp.optimize_layout(mp.generate_sis_mask(1, cap, seed=seed))
        cost = dally_eval.static_cost(ir)
        if cost is None:
            r = mp.evaluate_mask(ir)
            cost = r.cost
        results.append((cost, seed, cap))
    dt = time.perf_counter() - t0

    results.sort()
    print(f"scored {len(results)} candidates in {dt:.2f}s "
          f"({len(results) / dt:.0f} cand/s end-to-end incl. generation)")
    print("top 5 by cost:")
    for cost, seed, cap in results[:5]:
        print(f"  cost {cost:>10,}  seed {seed:>3}  cap {cap}")

    if dally_eval.available():
        # isolate scoring: re-score the top candidate 100x both ways
        seed, cap = results[0][1], results[0][2]
        ir = mp.optimize_layout(mp.generate_sis_mask(1, cap, seed=seed))
        t0 = time.perf_counter()
        for _ in range(100):
            dally_eval.static_cost(ir)
        rust_dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(5):
            mp.evaluate_mask(ir)
        py_dt = (time.perf_counter() - t0) / 5
        print(
            f"scoring isolation: rust {rust_dt / 100 * 1000:.2f} ms/ir vs "
            f"python ~{py_dt * 1000:.0f} ms/ir "
            f"(~{py_dt / (rust_dt / 100):.0f}x on the scoring step)"
        )


if __name__ == "__main__":
    main()
