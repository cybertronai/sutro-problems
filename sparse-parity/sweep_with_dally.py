"""Candidate static-cost sweep scored by the dally-eval Rust engine.

Demonstrates the in-repo inner loop: generate siswalk-family programs,
measure static read cost with dally-eval (via dally_eval.py), and fall
back to the Python parser if the binary is missing. Caps are reported
separately because they have different recovery behavior; this script
does not evaluate recovery.
Run from this directory:

    python3 sweep_with_dally.py            # 200 candidates
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

CAPS = (2, 3, 4)


def candidates(n: int):
    """(seed, cap) grid over the siswalk family."""
    if n < 1:
        raise ValueError("candidate count must be positive")
    for index in range(n):
        yield index // len(CAPS), CAPS[index % len(CAPS)]


def rank_by_cap(results):
    """Return static-cost rankings without comparing unlike caps."""
    grouped = {}
    for cost, seed, cap in results:
        grouped.setdefault(cap, []).append((cost, seed))
    return {
        cap: sorted(entries)
        for cap, entries in sorted(grouped.items())
    }


def python_static_cost(ir: str) -> int:
    """Closest Python equivalent to dally-eval's parse/static-cost path."""
    return mp._compile_ir(ir, mp.OP_CAP)[1]


def main() -> None:
    n = int(os.environ.get("DALLY_EVAL_CANDS", "200"))
    use_rust = dally_eval.available()
    engine = "dally-eval (rust)" if use_rust else "python parser fallback"
    print(f"engine: {engine}; candidates: {n}")
    print("static cost only; recovery is not evaluated; rankings are per cap")

    t0 = time.perf_counter()
    results = []
    for seed, cap in candidates(n):
        ir = mp.optimize_layout(mp.generate_sis_mask(1, cap, seed=seed))
        cost = dally_eval.static_cost(ir)
        if cost is None:
            cost = python_static_cost(ir)
        results.append((cost, seed, cap))
    dt = time.perf_counter() - t0

    print(f"scored {len(results)} candidates in {dt:.2f}s "
          f"({len(results) / dt:.1f} cand/s end-to-end incl. generation)")
    for cap, entries in rank_by_cap(results).items():
        print(f"cap {cap}: top {min(5, len(entries))} by static cost")
        for cost, seed in entries[:5]:
            print(f"  cost {cost:>10,}  seed {seed:>3}")

    if use_rust:
        # Compare equivalent parse/static-cost work in both implementations.
        _, seed, cap = results[0]
        ir = mp.optimize_layout(mp.generate_sis_mask(1, cap, seed=seed))
        rust_cost = dally_eval.static_cost(ir)
        python_cost = python_static_cost(ir)
        assert rust_cost == python_cost, (
            f"static cost mismatch: rust {rust_cost}, python {python_cost}"
        )
        t0 = time.perf_counter()
        for _ in range(100):
            dally_eval.static_cost(ir)
        rust_dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(5):
            python_static_cost(ir)
        py_dt = (time.perf_counter() - t0) / 5
        print(
            f"static-cost parsing: rust {rust_dt / 100 * 1000:.2f} ms/ir vs "
            f"python ~{py_dt * 1000:.0f} ms/ir "
            f"(~{py_dt / (rust_dt / 100):.0f}x)"
        )


if __name__ == "__main__":
    main()
